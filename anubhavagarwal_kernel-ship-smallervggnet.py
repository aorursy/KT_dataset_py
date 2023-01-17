# -*- coding: utf-8 -*-

"""

SmallerVGGNet model



@author: Anubhav Agarwal

"""

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation

from keras.layers.core import Flatten

from keras.layers.core import Dropout

from keras.layers.core import Dense

from keras import backend as K





class SmallerVGGNet:

    @staticmethod

    def build(width, height, depth, classes):

        model = Sequential()

        inputShape = (height, width, depth)

        chanDim = -1



        if K.image_data_format() == "channels_first":

            inputShape = (depth, height, width)

            chanDim = 1



        model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(3,3)))

        model.add(Dropout(0.25))



        model.add(Conv2D(64, (3,3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(64, (3,3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Dropout(0.25))



        model.add(Conv2D(128, (3,3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(128, (3,3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Dropout(0.25))



        model.add(Flatten())

        model.add(Dense(1024))

        model.add(Activation("relu"))

        model.add(BatchNormalization())

        model.add(Dropout(0.5))



        model.add(Dense(classes))

        model.add(Activation("sigmoid"))



        return model

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib

matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.preprocessing.image import img_to_array

from keras.utils import to_categorical

from keras.utils import plot_model

from sklearn.model_selection import train_test_split

#from model.smallervggnet import SmallerVGGNet

from keras.callbacks import ModelCheckpoint

import keras.backend as K

import matplotlib.pyplot as plt

import numpy as np

import argparse

import random

import cv2

import os

import glob

import pandas as pd

import tensorflow as tf
def category_0_to_n_minus1(x):

    x = x - 1

    return x



def f1(y_true, y_pred):

    y_pred = K.round(y_pred)

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)

    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)

    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)

    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)



    p = tp / (tp + fp + K.epsilon())

    r = tp / (tp + fn + K.epsilon())



    f1 = 2*p*r / (p+r+K.epsilon())

    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)

    return K.mean(f1)
train_data = pd.read_csv("../input/train/train.csv")

train_data.head(5)
train_data['category'] = train_data['category'].apply(category_0_to_n_minus1)
# initial parameters

epochs = 100

lr = 1e-3

batch_size = 64

img_dims = (96,96,3)



data = []

labels = []



random.seed(42)

random.shuffle(train_data.values)
# create groud-truth label from the image path

for index, row in train_data.iterrows():

    img_path = '../input/train/images/' + row['image']

    image = cv2.imread(img_path)

    

    image = cv2.resize(image, (img_dims[0],img_dims[1]))

    image = img_to_array(image)

    data.append(image)



    label = row['category']

        

    labels.append([label])
# pre-processing

data = np.array(data, dtype="float") / 255.0

labels = np.array(labels)



# split dataset for training and validation

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2,

                                                  random_state=42)

trainY = to_categorical(trainY, num_classes=5)

testY = to_categorical(testY, num_classes=5)



# augmenting datset 

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,

                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,

                         horizontal_flip=True, fill_mode="nearest")



# build model

model = SmallerVGGNet.build(width=img_dims[0], height=img_dims[1], depth=img_dims[2],

                            classes=5)



# compile the model

opt = Adam(lr=lr, decay=lr/epochs)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy', f1])

checkpoint = ModelCheckpoint('ship_detection.model', monitor='val_f1', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]

# train the model

H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),

                        validation_data=(testX,testY),

                        steps_per_epoch=len(trainX) // batch_size,

                        epochs=epochs, callbacks=callbacks_list, verbose=1)