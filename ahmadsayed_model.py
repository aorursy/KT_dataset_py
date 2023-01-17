import numpy as np

import pandas as pd

import json

import sys

from skimage.io import imread

from matplotlib import pyplot as plt

from keras import models

from keras.optimizers import SGD

import keras.models as models

from keras.layers import add

from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute

from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D

from keras.layers.normalization import BatchNormalization

from keras import backend as K

import cv2

import json

from tqdm import tqdm

import os



img_w = 256

img_h = 256

n_labels = 12

kernel = 3

np.random.seed(7) # 0bserver07 for reproducibility

print("Done")
encoding_layers = [

    Convolution2D(64, kernel, border_mode='same', input_shape=( img_h, img_w, 3)),

    BatchNormalization(),

    Activation('relu'),

    Convolution2D(64, kernel, border_mode='same'),

    BatchNormalization(),

    Activation('relu'),

    MaxPooling2D(),



    Convolution2D(128, kernel, kernel, border_mode='same'),

    BatchNormalization(),

    Activation('relu'),

    Convolution2D(128, kernel, kernel, border_mode='same'),

    BatchNormalization(),

    Activation('relu'),

    MaxPooling2D(),



    Convolution2D(256, kernel, kernel, border_mode='same'),

    BatchNormalization(),

    Activation('relu'),

    Convolution2D(256, kernel, kernel, border_mode='same'),

    BatchNormalization(),

    Activation('relu'),

    Convolution2D(256, kernel, kernel, border_mode='same'),

    BatchNormalization(),

    Activation('relu'),

    MaxPooling2D(),



    Convolution2D(512, kernel, kernel, border_mode='same'),

    BatchNormalization(),

    Activation('relu'),

    Convolution2D(512, kernel, kernel, border_mode='same'),

    BatchNormalization(),

    Activation('relu'),

    Convolution2D(512, kernel, kernel, border_mode='same'),

    BatchNormalization(),

    Activation('relu'),

    MaxPooling2D(),



    Convolution2D(512, kernel, kernel, border_mode='same'),

    BatchNormalization(),

    Activation('relu'),

    Convolution2D(512, kernel, kernel, border_mode='same'),

    BatchNormalization(),

    Activation('relu'),

    Convolution2D(512, kernel, kernel, border_mode='same'),

    BatchNormalization(),

    Activation('relu'),

    MaxPooling2D(),

]



autoencoder = models.Sequential()

autoencoder.encoding_layers = encoding_layers



for l in autoencoder.encoding_layers:

    autoencoder.add(l)

    print(l.input_shape,l.output_shape,l)



decoding_layers = [

    UpSampling2D(),

    Convolution2D(512, kernel, kernel, border_mode='same'),

    BatchNormalization(),

    Activation('relu'),

    Convolution2D(512, kernel, kernel, border_mode='same'),

    BatchNormalization(),

    Activation('relu'),

    Convolution2D(512, kernel, kernel, border_mode='same'),

    BatchNormalization(),

    Activation('relu'),



    UpSampling2D(),

    Convolution2D(512, kernel, kernel, border_mode='same'),

    BatchNormalization(),

    Activation('relu'),

    Convolution2D(512, kernel, kernel, border_mode='same'),

    BatchNormalization(),

    Activation('relu'),

    Convolution2D(256, kernel, kernel, border_mode='same'),

    BatchNormalization(),

    Activation('relu'),



    UpSampling2D(),

    Convolution2D(256, kernel, kernel, border_mode='same'),

    BatchNormalization(),

    Activation('relu'),

    Convolution2D(256, kernel, kernel, border_mode='same'),

    BatchNormalization(),

    Activation('relu'),

    Convolution2D(128, kernel, kernel, border_mode='same'),

    BatchNormalization(),

    Activation('relu'),



    UpSampling2D(),

    Convolution2D(128, kernel, kernel, border_mode='same'),

    BatchNormalization(),

    Activation('relu'),

    Convolution2D(64, kernel, kernel, border_mode='same'),

    BatchNormalization(),

    Activation('relu'),



    UpSampling2D(),

    Convolution2D(64, kernel, kernel, border_mode='same'),

    BatchNormalization(),

    Activation('relu'),

    Convolution2D(n_labels, 1, 1, border_mode='valid'),

    BatchNormalization(),

]

autoencoder.decoding_layers = decoding_layers

for l in autoencoder.decoding_layers:

    autoencoder.add(l)



autoencoder.add(Reshape((n_labels, img_h * img_w)))

autoencoder.add(Permute((2, 1)))

autoencoder.add(Activation('softmax'))

autoencoder.summary()
def normalized(rgb):

    #return rgb/255.0

    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3), np.float32)



    b=rgb[:,:,0]

    g=rgb[:,:,1]

    r=rgb[:,:,2]



    norm[:,:,0]=cv2.equalizeHist(b)

    norm[:,:,1]=cv2.equalizeHist(g)

    norm[:,:,2]=cv2.equalizeHist(r)



    return norm

def one_hot_it(labels):

    x = np.zeros([img_w,img_h,12])

    for i in range(img_w):

        for j in range(img_h):

            x[i,j,labels[i][j]]=1

    return x

import matplotlib.pyplot as plt

%matplotlib inline
data = []

label = []

for i in tqdm(os.listdir("../input/first_part")):

    sub_folder_path = os.path.join("../input/first_part", i)

    for j in tqdm(os.listdir(sub_folder_path)):

        sub_sub_folder = os.path.join(sub_folder_path, j)

        for ij in os.listdir(sub_sub_folder):

            print(ij)

            

            if "_seg.png" in ij:

                img_path = os.path.join(sub_sub_folder, ij)

                img = cv2.imread(img_path)

                if img is not None:

                    img = cv2.resize(img, (img_w, img_h))

                    data.append([ij.split('_seg.png')[0], np.rollaxis(normalized(img),2)])

                    plt.imshow(img)

                    plt.show()

            if ".jpg"  in ij:

                img_path = os.path.join(sub_sub_folder, ij)

                img = cv2.imread(img_path)

                if img is not None:

                    img = cv2.resize(img, (img_w, img_h))

                    label.append([ij.split(".jpg")[0], np.rollaxis(normalized(img),2)])

                    plt.imshow(img)

                    plt.show()

            else:

                continue

        break

    break

            

    
dataArray = np.array(data)

labelArray = np.array(label)



def takeSecond(elem):

    return elem[0]



sortData = np.array(sorted(dataArray, key=takeSecond))



sortLabel = np.array(sorted(labelArray, key=takeSecond))
data = []

for i in range(len(sortData)):

    data.append(np.array([sortData[i, 1], sortLabel[i, 1]]))

np.array(data)

print(np.array(data).shape)
for i in range(len(data)):

    plt.imshow(data[i,0])

    plt.show()

    plt.imshow(data[i,1])

    plt.show()
# def load_data(mode):

#     data = []

#     label = []

#     with open(DataPath + mode +'.txt') as f:

#         txt = f.readlines()

#         txt = [line.split(' ') for line in txt]

#     for i in range(len(txt)):

#         data.append(np.rollaxis(normalized(cv2.imread(os.getcwd() + txt[i][0][7:])),2))

#         label.append(one_hot_it(cv2.imread(os.getcwd() + txt[i][1][7:][:-1])[:,:,0]))

#         print('.',end='')

#     return np.array(data), np.array(label)