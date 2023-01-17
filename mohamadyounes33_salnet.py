from keras.optimizers import SGD

import h5py

import numpy as np

import keras

from keras import backend as K

from keras.optimizers import SGD

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.applications.vgg16 import VGG16

from keras.layers import Input, Activation, ReLU, Layer, MaxPooling2D, Deconvolution2D, Conv2D

from keras.layers import Activation, BatchNormalization, Input

from keras.layers.convolutional import Convolution2D, UpSampling2D

from keras.models import Model

from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler

import pandas as pd

from keras.preprocessing import image

import matplotlib.pyplot as plt

%matplotlib inline

import cv2

from PIL import Image

from scipy import ndimage

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

keras.backend.set_image_data_format("channels_first")
def norm_weights(n):

    r = n / 2.0

    xs = np.linspace(-r, r, n)

    x, y = np.meshgrid(xs, xs)

    w = np.exp(-0.5*(x**2 + y**2))

    w /= w.sum()

    return w
def deconv(nb_filter, size, name):

    upsample = UpSampling2D(size=(size, size))

    s = 2 * size + 1

    w = norm_weights(s)[np.newaxis, np.newaxis, :, :]

    conv = Convolution2D(

        nb_filter, s, s,

        name=name,

        activation='linear',

        bias=False,

        border_mode='same',

        weights=[w])

    return lambda x: conv(upsample(x))
# This is the work of https://github.com/massens/salnet-keras/blob/master/model1_salnet_keras.py

def get_model():



    input_tensor = Input(shape=(3, 240, 320)) 

    base_model = VGG16(False, weights='imagenet', input_tensor=input_tensor)



    x = base_model.get_layer('block4_conv2').output

    x = Convolution2D(512, 5, 5, activation='relu', border_mode='same', name='conv4')(x)

    x = Convolution2D(512, 5, 5, activation='relu', border_mode='same', name='conv5')(x)

    x = Convolution2D(256, 7, 7, activation='relu', border_mode='same', name='conv6')(x)

    x = Convolution2D(128, 11, 11, activation='relu', border_mode='same', name='conv7')(x)

    x = Convolution2D(32 , 11, 11, activation='relu', border_mode='same', name='conv8')(x)

    x = Convolution2D(1 , 13, 13, activation='relu', border_mode='same', name='conv9')(x)

    x = UpSampling2D(size=(8,8))(x)

    x = Convolution2D(1, 9, 9, name='deconv', bias=False, border_mode='same', activation='linear')(x)

    #x = deconv(1,4, 'deconv')(x)

    #output = Deconvolution2D(1, 8, 8, bias=False, subsample=(8,8), name = 'deconv1')(x)

    output = Activation('sigmoid')(x)



    model = Model(input=input_tensor, output=output)



    # for layer in base_model.layers:

    #     w = layer.get_weights()

    #     if len(w) > 0 : # Is convolutional

    #         w[1] = w[1] * (1.0/150)

    #         layer.set_weights(w)



    sgd = SGD(lr=1.3e-7)

    model.compile(optimizer=sgd, loss='binary_crossentropy')

    return model
# My model

def get_model():



    input_tensor = Input(shape=(3, 240, 320)) 

    x = Conv2D(96, kernel_size=7, activation = 'relu', strides=1,padding='same', name='conv1')(input_tensor)

    #x = Activation('relu', name='relu1')(x)

    x = BatchNormalization(name='norm1')(x)

    x = MaxPooling2D(pool_size=(3,3), name='pool1',padding='same', strides=(2,2))(x)

    x = Conv2D(256, kernel_size= 5, activation = 'relu', strides=(1,1),padding='same', name='conv2')(x)

    x = MaxPooling2D(pool_size=(3,3), name='pool2', strides=(2,2))(x)

    x = Conv2D(512, kernel_size=3, activation = 'relu', strides=(1,1),padding='same', name='conv3')(x)

    x = Conv2D(512, kernel_size= 5, activation = 'relu', strides=(1,1),padding='same', name='conv4')(x)

    x = Conv2D(512, kernel_size= 5, activation = 'relu', strides=(1,1),padding='same', name='conv5')(x)

    x = Conv2D(256, kernel_size= 7, activation = 'relu', strides=(1,1),padding='same', name='conv6')(x)

    x = Conv2D(128, kernel_size= 11, activation = 'relu', strides=(1,1),padding='same', name='conv7')(x)

    x = Conv2D(32, kernel_size=11, activation = 'relu', strides=(1,1),padding='same', name='conv8')(x)

    x = Conv2D(1, kernel_size= 13, activation = 'relu', strides=(1,1),padding='same', name='conv9')(x)

    #output = deconv(1, 8, name = 'deconv1')(x)

    #output = Deconvolution2D(8, 1 , name = 'deconv1')(x)

    x = Deconvolution2D(1, 8, 8, output_shape = (None, 1, 240, 320), bias=False, subsample=(4,4), name = 'deconv1')(x)

    output = Activation('sigmoid')(x)

    model = Model(input=input_tensor, output=output)



    sgd = SGD(lr=1.3e-7)

    model.compile(optimizer=sgd, loss='binary_crossentropy')

    return model
model = get_model()
model.summary()
model.load_weights("/kaggle/input/saliency/salnet.h5", by_name=True)
im = Image.open("/kaggle/input/images-test/black-cat.jpg")

plt.imshow(im)
img = load_img("/kaggle/input/images-test/black-cat.jpg")

img = img.resize((320, 240))

x = img_to_array(img)

x = x.reshape((1,3,240,320))

plt.imshow(x[0][0])
pred = model.predict(x)
blured= ndimage.gaussian_filter(pred[0], sigma=3)

sal_map = blured

sal_map -= np.min(sal_map)

sal_map /= np.max(sal_map)

print(sal_map.shape)

plt.imshow(sal_map[0])