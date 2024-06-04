import streamlit as st
import os
st.title("STYLE TRANSFER")
st.header("Select an Image:")
img=st.file_uploader("image")
st.header("Select an Art or Style:")
art=st.file_uploader("art")
save1_dir="uploaded images"
save2_dir="uploaded art"
if img is not None and art is not None:
    imgname = img.name
    artname=art.name
    save1_path = os.path.join(save1_dir, imgname)
    if not os.path.exists(save1_dir):
        os.makedirs(save1_dir)
    with open(save1_path, "wb") as f:
        f.write(img.getbuffer())
    save2_path = os.path.join(save2_dir, artname)
    if not os.path.exists(save2_dir):
        os.makedirs(save2_dir)
    with open(save2_path, "wb") as f:
        f.write(art.getbuffer())

    st.success(f"Image saved to: {save1_path}")
    st.success(f"Image saved to: {save2_path}")
    img="uploaded images/"+imgname
    art="uploaded art/"+artname
    #st.image(img)
    #st.image(art)


    import tensorflow as tf
    import time
    # Load compressed models from tensorflow_hub
    os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
    import matplotlib as mpl
    mpl.rcParams['figure.figsize'] = (12, 12)
    mpl.rcParams['axes.grid'] = False


    def load_img(path_to_img):
        max_dim = 512
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img


    content_image = load_img(img)
    style_image = load_img(art)
    if st.button("GENERATE"):
        import tensorflow_hub as hub
        hub_model = hub.load('model')
        with st.spinner("Wait for a few seconds,creating art..."):
            stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
            stylized_image = stylized_image.numpy()
        if stylized_image.any():
            st.success("Image Generated!")
            col1,col2,col3=st.columns(3)
            col1.subheader("Selected image:")
            col1.image(img)
            col2.subheader("Selected style:")
            col2.image(art)
            col3.subheader("style transfered image:")
            col3.image(stylized_image)

