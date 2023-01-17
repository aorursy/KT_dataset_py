import os

import sys

import scipy

from matplotlib.pyplot import imshow

from PIL import Image

import numpy as np

import tensorflow as tf

import keras

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

from keras.preprocessing import image

%matplotlib inline

print(os.listdir("../input/"))
content_image_location = '../input/City.jpg'

style_image_location = "../input/Clouds.jpg"
content_image = scipy.misc.imread(content_image_location)

imshow(content_image)

content_image.shape
style_image = scipy.misc.imread(style_image_location)

imshow(style_image)

style_image.shape
model = VGG16(include_top=True)

model.summary()
img = image.load_img(content_image_location, target_size=(224,224))

img
x = np.expand_dims(image.img_to_array(img), axis=0)

x.shape
x = preprocess_input(x)

x.shape
features = model.predict(x)

features.shape
decode_predictions(features, top=10)[0]