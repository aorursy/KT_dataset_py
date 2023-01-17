# Generic Libraries

import numpy as np

import os, gc , warnings, time

warnings.filterwarnings("ignore")



# Imaging & Ploting Libraries

from PIL import Image

import matplotlib.pyplot as plt

%matplotlib inline



# TensorFlow Libraries

import tensorflow as tf

import tensorflow_hub as hub
# Download Image from internet

!wget 'https://lh4.googleusercontent.com/-Anmw5df4gj0/AAAAAAAAAAI/AAAAAAAAAAc/6HxU8XFLnQE/photo.jpg64' -O original.jpg
# Model URL & Image Path

model_url = 'https://tfhub.dev/captain-pool/esrgan-tf2/1'

image_path = './original.jpg'
# Pre-Process Image

def preprocess_image(image_path):

    hr_image = tf.image.decode_image(tf.io.read_file(image_path))

    

    # If PNG, remove the alpha channel. The model only supports images with 3 color channels.

    if hr_image.shape[-1] == 4:

        hr_image = hr_image[...,:-1]

        

    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4

    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])

    hr_image = tf.cast(hr_image, tf.float32)

    return tf.expand_dims(hr_image, 0)



# Saves unscaled Tensor Images

def save_image(image, filename):

    if not isinstance(image, Image.Image):

        image = tf.clip_by_value(image, 0, 255)

        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())

    image.save("%s.jpg" % filename)

    print("Saved as %s.jpg" % filename)

    

# Plots images from image tensors

def plot_image(image, title=""):

    image = np.asarray(image)

    image = tf.clip_by_value(image, 0, 255)

    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())

    plt.imshow(image)

    plt.axis("off")

    plt.title(title)
# Load Model

model = hub.load(model_url)
# Load, Plot & Save Original Image

lr_image = preprocess_image(image_path)

plot_image(tf.squeeze(lr_image), title="Low Resolution")

#save_image(tf.squeeze(lr_image), filename="Original Image")
# Enhance Resolution

start = time.time()           # start timer

hr_image = model(lr_image)

hr_image = tf.squeeze(hr_image)

end = time.time()           # end timer



print("Time Taken: %f sec" % (end - start))
# Plotting High Resolution Image

plot_image(tf.squeeze(hr_image), title="High Resolution")