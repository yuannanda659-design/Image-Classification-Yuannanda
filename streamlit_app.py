# Converted from Load_Image_Classification.ipynb
# Streamlit App
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import tempfile
import os

st.title('Klasifikasi Anggur')


MODEL_PATH = "model_anggur.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

class_names = ['merah', 'hijau']

def prediksi_gambar(path_gambar):
    img = image.load_img(path_gambar, target_size=(227,227))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    hasil = model.predict(img_array, verbose=0)
    prediksi = np.argmax(hasil)

    return hasil, class_names[prediksi]

uploaded_file = st.file_uploader(
    "Upload gambar",
    type=["jpg","jpeg","png","webp"]
)

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img.save(tmp.name)
        hasil, prediksi = prediksi_gambar(tmp.name)

    st.success(f"Prediksi: {prediksi}")
    st.write(hasil)

    os.unlink(tmp.name)
