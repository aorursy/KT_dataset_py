from __future__ import print_function

import numpy as np

import scipy.io

import scipy.misc

import pandas as pd

import os

import cv2

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow

import seaborn as sns

import keras

from keras.preprocessing.image import load_img, img_to_array

from PIL import Image

import tensorflow as tf



from keras.applications.vgg16 import VGG16

from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input

from keras import backend

from keras.models import Model

from keras.applications.vgg16 import VGG16

from scipy.optimize import fmin_l_bfgs_b
def image(image_path,target_size):

    img = load_img(image_path, target_size=target_size)

    img=img_to_array(img)

    return img
base_image_path="../input/dataset12/model2.jpg"

style_image_path="../input/dataset12/sketch.jpg"

target_size=(437,450)

base_image=image(base_image_path,target_size)

style_image=image(style_image_path,target_size)
print("Base image shape:{} and Style image shape:{}".format(base_image.shape, style_image.shape))
base_image_plot = load_img(base_image_path, target_size=target_size)

style_image_plot = load_img(style_image_path, target_size=target_size)

fig=plt.figure(figsize=(7,7))

ax=plt.subplot(1,2,1)

ax.set_title("Base Image")

imshow(base_image_plot)

ax1=plt.subplot(1,2,2)

ax1.set_title("Style Image")

imshow(style_image_plot)

plt.show()
def preprocess(img):

    img = img.copy()                   

    img = np.expand_dims(img, axis=0) 

    return keras.applications.vgg16.preprocess_input(img)
def deprocess(img):

    img = img.copy()                   

    img = img[0]                        

    img[:, :, 0] += 103.939           

    img[:, :, 1] += 116.779             

    img[:, :, 2] += 123.68             

    img = img[:, :, ::-1]              

    img = np.clip(img, 0, 255)         

    return img.astype('uint8')
def inputs(original_img, style_img):

    original_input   = tf.constant(preprocess(original_img))

    style_input     = tf.constant(preprocess(style_img))

    generated_input = tf.placeholder(tf.float32, original_input.shape)

    return original_input, style_input, generated_input
original_input, style_input, generated_input = inputs(base_image, style_image)

input_tensor = tf.concat([original_input, style_input, generated_input], axis=0)

input_tensor.shape
vgg16_model = keras.applications.vgg16.VGG16(

    weights="imagenet",

    input_tensor=input_tensor, 

    include_top=False)

vgg16_model.summary()
vgg16_layer_dict = {layer.name:layer for layer in vgg16_model.layers}

for key,val in vgg16_layer_dict.items():

    print("{} => {}".format(key,val))
def calculate_original_loss(layer_dict, original_layer_names):

    loss = 0

    for name in original_layer_names:

        layer = layer_dict[name]

        original_features = layer.output[0, :, :, :]  

        generated_features = layer.output[2, :, :, :] 

        loss += keras.backend.sum(keras.backend.square(generated_features - original_features))

    return loss / len(original_layer_names)
def gram_matrix(x):    

    features = keras.backend.batch_flatten(keras.backend.permute_dimensions(x, (2, 0, 1))) 

    gram = keras.backend.dot(features, keras.backend.transpose(features))

    return gram
def get_style_loss(style_features, generated_features, size):

    S = gram_matrix(style_features)

    G = gram_matrix(generated_features)

    channels = 3

    return keras.backend.sum(keras.backend.square(S - G)) / (4. * (channels**2) * (size**2))
def calculate_style_loss(layer_dict, style_layer_names, size):

    loss = 0

    for name in style_layer_names:

        layer = layer_dict[name]

        style_features     = layer.output[1, :, :, :] 

        generated_features = layer.output[2, :, :, :] 

        loss += get_style_loss(style_features, generated_features, size) 

    return loss / len(style_layer_names)
def calculate_variation_loss(x):

    row_diff = keras.backend.square(x[:, :-1, :-1, :] - x[:, 1:,    :-1, :])

    col_diff = keras.backend.square(x[:, :-1, :-1, :] - x[:,  :-1, 1:,   :])

    return keras.backend.sum(keras.backend.pow(row_diff + col_diff, 1.25))
original_loss = calculate_original_loss(vgg16_layer_dict, ['block5_conv2'])

style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1', 'block5_conv1']

style_loss = calculate_style_loss(vgg16_layer_dict, style_layers, 

                                  base_image.shape[0]*base_image.shape[1])

variation_loss = calculate_variation_loss(generated_input)
from tqdm import tqdm

loss = 0.5 * original_loss + 1.0 * style_loss + 0.1 * variation_loss

gradients = keras.backend.gradients(loss, generated_input)[0]

calculate = keras.backend.function([generated_input], [loss, gradients])

generated_data=preprocess(base_image)

for i in tqdm(range(250)):

    _, gradients_value = calculate([generated_data])

    generated_data -= gradients_value * 0.001
fig=plt.figure(figsize=(10,10))

ax=plt.subplot(1,3,1)

ax.set_title("Base Image")

imshow(base_image_plot)

ax1=plt.subplot(1,3,2)

ax1.set_title("Style Image")

imshow(style_image_plot)





generated_image01 = deprocess(generated_data)

ax1=plt.subplot(1,3,3)

ax1.set_title("Output Image")

imshow(cv2.cvtColor(generated_image01, cv2.COLOR_BGR2RGB))

plt.show()