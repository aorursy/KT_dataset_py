import IPython.display as display



import tensorflow as tf

import tensorflow_hub as hub



import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12,12)

mpl.rcParams['axes.grid'] = False



import numpy as np

import PIL.Image

import time

import functools

import random



import time

import os

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



notebookstart = time.time()
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')

style_DIR = "../input/gan-getting-started/monet_jpg/"

style_images = os.listdir(style_DIR)



landscape_DIR = "../input/landscape-pictures/"

landscape_images = os.listdir(landscape_DIR)
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



def imshow(image, ax, title=None):

    if len(image.shape) > 3:

        image = tf.squeeze(image, axis=0)



        ax.imshow(image)

        if title:

            ax.set_title(title)

            
for _ in range(35):

    landscape_i = landscape_DIR + random.choice(landscape_images)

    style_i = style_DIR + random.choice(style_images)



    content_image = load_img(landscape_i)

    style_image = load_img(style_i)



    f, ax = plt.subplots(1,3,figsize=[22,10])

    imshow(image=style_image, ax=ax[0], title="Random Style Image - {}".format(landscape_i.split("/")[-1]))

    imshow(image=content_image, ax=ax[1], title="Random Content Image - {}".format(style_i.split("/")[-1]))



    stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]

    imshow(image=stylized_image, ax=ax[2], title='Stylized Image')

    

    ax[0].axis('off')

    ax[1].axis('off')

    ax[2].axis('off')

    

    plt.tight_layout(pad=1)

    plt.show()
print("Notebook Runtime: %0.2f Hours"%((time.time() - notebookstart)/60/60))