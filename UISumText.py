import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer (cached to improve performance)
@st.cache_resource
def load_model():
    model_path = "migz117/T5-Abstractive"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

# Generate summary function
def generate_summary(input_text, model, tokenizer):
    try:
        # Prepare input
        inputs = tokenizer(input_text, return_tensors="pt", 
                           truncation=True, 
                           padding=True, 
                           max_length=512).to(device)
        
        # Generate summary
        summary_ids = model.generate(**inputs, 
                                     max_length=200, 
                                     num_return_sequences=1,
                                     do_sample=False)
        
        # Decode summary
        abstractive_summary = tokenizer.decode(summary_ids[0], 
                                               skip_special_tokens=True)
        
        return abstractive_summary.strip()
    
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None

# Main Streamlit app
def main():
    # Page configuration
    st.set_page_config(page_title="SumText - Online Summarize", layout="centered")
    
    # Load model
    model, tokenizer = load_model()

    # UI Elements
    st.image("Logo AI.png", width=200)
    st.markdown("<h1 style='text-align: center; color: #00C2C2;'>SumText</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>ONLINE SUMMARIZE</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Dapatkan ringkasan teks yang jelas dan informatif dengan SumText<br>AI peringkas yang membantu anda menyederhanakan informasi secara cepat dan akurat.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Text input
    st.markdown("**Masukkan teks Anda di sini...**")
    input_text = st.text_area("", height=200, max_chars=2000, placeholder="Tulis teks di sini...", label_visibility="collapsed")
    
    # Character count
    st.markdown(
        f"""
        <p style="font-size: 16px; color: #00C2C2;">
            ID 
            <span style="display: inline-block; width: 0; height: 0; border-left: 5px solid transparent; border-right: 5px solid transparent; border-bottom: 10px solid #00C2C2; margin-left: 5px; margin-right: 5px;"></span>
            ({len(input_text)}/2000) Karakter
        </p>
        """, unsafe_allow_html=True
    )

    # Summarize button
    if st.button("Summarize"):
        if input_text.strip():
            with st.spinner("Sedang memproses..."):  # Tambahkan spinner di sini
                # Generate summary using the model
                summary = generate_summary(input_text, model, tokenizer)
            
            if summary:
                # Display summary
                output_html = f"""
                    <div style="padding: 10px; border: 2px solid #00C2C2; border-radius: 10px; background-color: #F0F8FF; height: 200px; overflow-y: auto;">
                    <p style="margin: 0; font-size: 14px; color: black;">{summary}</p>
                    </div>
                """
                summary_length = len(summary)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Hasil Ringkasan**")
                st.markdown(output_html, unsafe_allow_html=True)
                
                # Character count for summary
                st.markdown(
                    f"""
                    <p style="font-size: 16px; color: #00C2C2;">
                        ID 
                        <span style="display: inline-block; width: 0; height: 0; border-left: 5px solid transparent; border-right: 5px solid transparent; border-bottom: 10px solid #00C2C2; margin-left: 5px; margin-right: 5px;"></span>
                        ({summary_length}/2000) Karakter
                    </p>
                    """, unsafe_allow_html=True
                )
                
                # Copy and Download buttons
                col1, col_center, col2 = st.columns([1, 2, 1])
                with col1:
                    st.components.v1.html(f"""
                    <script>
                    function copyToClipboard() {{
                        navigator.clipboard.writeText(`{summary}`).then(function() {{
                            alert("Ringkasan berhasil disalin!");
                        }}, function(err) {{
                            console.error('Tidak dapat menyalin teks: ', err);
                        }});
                    }}
                    </script>
                    <button onclick="copyToClipboard()" style="width: 100%; background-color: #00C2C2; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer;">
                        Copy to Clipboard
                    </button>
                    """, height=50)
                
                with col2:
                    st.download_button(
                        label="Download",
                        data=summary,
                        file_name="ringkasan.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            else:
                st.error("Gagal membuat ringkasan.")
        else:
            st.warning("Masukkan teks untuk diringkas!")


    # Footer
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 12px;'>© 2024 SumText. All rights reserved. | Privacy Policy | Terms of Service</p>", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
