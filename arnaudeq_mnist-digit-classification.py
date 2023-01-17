import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from keras.models import Sequential
from __future__ import division,print_function

import math, os, json, sys, re

import _pickle as pickle

from glob import glob

import numpy as np

from matplotlib import pyplot as plt

from operator import itemgetter, attrgetter, methodcaller

from collections import OrderedDict

import itertools

from itertools import chain

from imp import reload



import pandas as pd

import PIL

from PIL import Image

from numpy.random import random, permutation, randn, normal, uniform, choice

from numpy import newaxis

import scipy

from scipy import misc, ndimage

from scipy.ndimage.interpolation import zoom

from scipy.ndimage import imread

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import OneHotEncoder

from sklearn.manifold import TSNE

from sklearn.preprocessing import LabelBinarizer

from IPython.lib.display import FileLink

from keras.wrappers.scikit_learn import KerasClassifier



import tensorflow as tf

import keras

from keras import backend as K

from keras.utils.data_utils import get_file

from keras.utils import np_utils

from keras.utils.np_utils import to_categorical

from keras.models import Sequential, Model

from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional

from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU

from keras.layers.core import Flatten, Dense, Dropout, Lambda

from keras.regularizers import l2, l1

from keras.layers.normalization import BatchNormalization

from keras.layers.pooling import *

from keras.optimizers import SGD, RMSprop, Adam

from keras.metrics import categorical_crossentropy, categorical_accuracy

from keras.layers.convolutional import *

from keras.preprocessing import image, sequence

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.resnet50 import ResNet50

from keras.applications.vgg16 import VGG16

from keras.applications.inception_v3 import InceptionV3

from keras import applications

from scipy.misc import imread, imresize, imsave

from string import Template

from bs4 import BeautifulSoup



def load_array(fname):

    return bcolz.open(fname)[:]



def extract_img_info(xml):

    img_info = dict()

    img_info["size"] = [int(n) for n in xml.size.stripped_strings][:-1]

    img_info["filename"] = xml.find_all("filename")[0].text

    bboxes = dict()

    tmp = dict()

    for obj in xml.find_all("object"):

        img_info["tag"] = obj.find_all("name")[0].text

        for n in obj.find_all("bndbox"):

            tmp["xmin"] = int(n.find_all("xmin")[0].text)

            tmp["ymin"] = int(n.find_all("ymin")[0].text)

            tmp["width"] = int(n.find_all("xmax")[0].text) - tmp['xmin']

            tmp["height"] = int(n.find_all("ymax")[0].text) - tmp['ymin']

    img_info["bboxes"] = tmp

    return img_info



# below you'll need to provide the path of the folder where all the files are listed

def get_list_annotations(path):

    listing = sorted(os.listdir(path))

    annotations = []

    for file in listing:

        xml = open(path + file,'r')

        xml = BeautifulSoup(xml, "lxml")

        img_info = extract_img_info(xml)

        annotations.append(img_info)

    return(annotations)



def dict_to_pd(file):

    d = file

    df = pd.DataFrame([],columns=['filename','img_width','img_height','bb_x','bb_y','bb_width','bb_height'])

    for i in range(0,len(d)):

        to_add = pd.DataFrame([[d[i]['filename'],

                       d[i]['size'][0], 

                       d[i]['size'][1],

                       d[i]['bboxes']['xmin'],

                       d[i]['bboxes']['ymin'],

                       d[i]['bboxes']['width'], 

                       d[i]['bboxes']['height']]], 

                      columns=['filename','img_width','img_height','bb_x','bb_y','bb_width','bb_height'])

        to_add.set_index([[i]],inplace=True)

        df = df.append(to_add)

    return(df)



def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True, batch_size=4, class_mode='categorical',

                target_size=(224,224)):

    return gen.flow_from_directory(dirname, target_size=target_size,

            class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)



def get_data(path, target_size=(224,224)):

    batches = get_batches(path, shuffle=False, batch_size=1, class_mode=None, target_size=target_size)

    return np.concatenate([batches.next() for i in range(batches.samples)])
%matplotlib inline

from __future__ import division, print_function

from keras.backend.tensorflow_backend import set_session

from keras.datasets import mnist
config = tf.ConfigProto()

config.gpu_options.per_process_gpu_memory_fraction = 0.5

set_session(tf.Session(config=config))
import numpy as np
mnist = np.load("../input/mnist.npz")
(x_train,y_train),(x_test,y_test) = (mnist["x_train"], mnist["y_train"]), (mnist["x_test"], mnist["y_test"])



fig, ax = plt.subplots(nrows=2,ncols=10,figsize=(20,5))

ax = ax.ravel()



for i in range(20):

    ax[i].imshow(x_train[i],cmap = plt.get_cmap('gray'))
x_train.shape
plt.imshow(x_train[3,:28,:28], cmap='gray')
plt.imshow(x_train[-1,:,:], cmap='gray')
y_train
# obtaining a vector of 784 for the 28x28 images

num_pixels = x_train.shape[1] * x_train.shape[2]

x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')

x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')



# normalizing between 0 and 1

x_train = x_train / 255

x_test = x_test / 255



y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)

print('Number of categories:',y_train.shape[1])
def neuron_vanilla():

    model = Sequential()

    

    model.add(Dense(10,activation='softmax',input_dim=num_pixels))

    

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    return model



model = neuron_vanilla()

model.summary()
model.fit(x_train,y_train, validation_data =(x_test,y_test),epochs=9,batch_size=64, verbose=True)
x_test.shape
img = x_test[0].reshape(1, -1)
img.shape
plt.imshow(img.reshape(28, 28),cmap = plt.get_cmap('gray'))
y_test[0]
model.predict(img)
np.argmax(_)
fails = []

for index in range(len(x_test)):

    img = x_test[index].reshape(1, -1)

    if int(np.argmax(model.predict(img))) != int(np.argmax(y_test[index])):

        fails.append(index)
len(fails)
x_train[0].shape
index_to_check = fails[90]
plt.imshow(x_test[index_to_check].reshape(28,28),cmap = plt.get_cmap('gray'))
np.argmax(y_test[index_to_check])
np.argmax(model.predict(x_test[index_to_check].reshape(1, -1)))
scores = model.evaluate(x_test,y_test,verbose=0)

print('loss: ', scores[0],'- accuracy: ', scores[1])
def MLP():

    model = Sequential()

    

    model.add(Dense(256,activation='relu',input_dim=num_pixels))

    model.add(Dense(256,activation='relu'))

    

    model.add(Dense(10,activation='softmax'))

    

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    return model



model = MLP()

model.summary()
model.fit(x_train,y_train, validation_data =(x_test,y_test),epochs=10,batch_size=64, verbose=True)
scores = model.evaluate(x_test,y_test,verbose=0)

print('loss: ', scores[0],'- accuracy: ', scores[1])
def MLP_b():

    model = Sequential()



    model.add(Dense(256,activation='relu', input_dim=num_pixels))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

              

    model.add(Dense(256,activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

    

    model.add(Dense(10,activation='softmax'))

    

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    return model



model = MLP_b()

model.summary()
model.fit(x_train,y_train, validation_data =(x_test,y_test),epochs=15,batch_size=64, verbose=True)
scores = model.evaluate(x_test,y_test,verbose=0)

print('loss: ', scores[0],'- accuracy: ', scores[1])
(x_train,y_train),(x_test,y_test) = (mnist["x_train"], mnist["y_train"]), (mnist["x_test"], mnist["y_test"])
# reshaping the images for the conv2D (channels last)

x_train = x_train.reshape(x_train.shape[0],28,28,1).astype('float32')

x_test = x_test.reshape(x_test.shape[0],28,28,1).astype('float32')



# one hot encode for the categories

y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)
x_train.shape
batch_size = 64

train_samples = x_train.shape[0]

validation_samples = x_test.shape[0]



data_gen_train = ImageDataGenerator(rescale=1./255,

                              rotation_range=8,

                              width_shift_range=0.1,

                              height_shift_range=0.1,

                              shear_range=0.3,

                              zoom_range=0.1,

                              horizontal_flip=False)



data_gen_test = ImageDataGenerator(rescale=1./255)



train_generator = data_gen_train.flow(x_train, y_train, batch_size = batch_size)

validation_generator = data_gen_test.flow(x_test, y_test, batch_size = batch_size, shuffle=False)
def complex_conv_model():

    

    # creating the convnet model

    model = Sequential()

    

    # first convo block

    model.add(Conv2D(32,(5,5), activation='relu', input_shape = (28,28,1)))

    model.add(BatchNormalization())

    model.add(Conv2D(32,(5,5), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(BatchNormalization())



    # two convo block

    model.add(Conv2D(64,(3,3),activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64,(3,3),activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(BatchNormalization())

    

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

        

    model.add(Dense(128, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

              

    model.add(Dense(10,activation='softmax'))

    

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    return model



model = complex_conv_model()

model.summary()
model.fit_generator(train_generator,

        steps_per_epoch = train_samples // batch_size,

        epochs=10,

        validation_data = validation_generator,

        validation_steps = validation_samples // batch_size)