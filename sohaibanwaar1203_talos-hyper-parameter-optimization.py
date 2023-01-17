!pip install talos
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation

from keras.activations import relu, elu

from keras.layers.core import Flatten

from keras.layers.core import Dropout

from keras.layers.core import Dense

from keras import backend as K

import talos as ta

from keras.wrappers.scikit_learn import KerasClassifier

from keras.preprocessing.image import img_to_array

import matplotlib

matplotlib.use("Agg")

import PIL 

from keras import regularizers

from PIL import Image

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import SGD

import keras

import matplotlib.pyplot as plt

import numpy as np

import argparse

import random

import pickle

import cv2

import os



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/sign_mnist_train.csv')

test = pd.read_csv('../input/sign_mnist_test.csv')

labels = train['label'].values

train.drop('label', axis = 1, inplace = True)

images = train.values

images = np.array([np.reshape(i, (28, 28)) for i in images])

images = np.array([i.flatten() for i in images])

from sklearn.preprocessing import LabelBinarizer

label_binrizer = LabelBinarizer()

labels = label_binrizer.fit_transform(labels)
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3)
x_train = x_train / 255

x_test = x_test / 255

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,

                        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,

            horizontal_flip=True, fill_mode="nearest")



params = {'lr': (0.1, 0.01,1 ),

     'epochs': [10,5,15],

     'dropout': (0, 0.40, 0.8),

     'optimizer': ["Adam","Adagrad","sgd"],

     'loss': ["binary_crossentropy","mean_squared_error","mean_absolute_error"],

     'last_activation': ["softmax","sigmoid"],

     'activation' :["relu","selu","linear"],

     'clipnorm':(0.0,0.5,1),

     'decay':(1e-6,1e-4,1e-2),

     'momentum':(0.9,0.5,0.2),

     'l1': (0.01,0.001,0.0001),

     'l2': (0.01,0.001,0.0001),

     'No_of_CONV_and_Maxpool_layers':[1,2],

     'No_of_Dense_Layers': [2,3,4],

     'No_of_Units_in_dense_layers':[64,32],

     'Kernal_Size':[(3,3),(5,5)],

     'Conv2d_filters':[60,40,80,120],

     'pool_size':[(3,3),(5,5)],

     'padding':["valid","same"]

    }



def Talos_Model(X_train, y_train, X_test, y_test, params):

    #parameters defined

    lr = params['lr']

    epochs=params['epochs']

    dropout_rate=params['dropout']

    optimizer=params['optimizer']

    loss=params['loss']

    last_activation=params['last_activation']

    activation=params['activation']

    clipnorm=params['clipnorm']

    decay=params['decay']

    momentum=params['momentum']

    l1=params['l1']

    l2=params['l2']

    No_of_CONV_and_Maxpool_layers=params['No_of_CONV_and_Maxpool_layers']

    No_of_Dense_Layers =params['No_of_Dense_Layers']

    No_of_Units_in_dense_layers=params['No_of_Units_in_dense_layers']

    Kernal_Size=params['Kernal_Size']

    Conv2d_filters=params['Conv2d_filters']

    pool_size_p=params['pool_size']

    padding_p=params['padding']

    

    #model sequential

    model=Sequential()

    

    for i in range(0,No_of_CONV_and_Maxpool_layers):

        model.add(Conv2D(Conv2d_filters, Kernal_Size ,padding=padding_p))

        model.add(Activation(activation))

        model.add(MaxPooling2D(pool_size=pool_size_p,strides=(2,2)))

    

    

    model.add(Flatten())

    

    for i in range (0,No_of_Dense_Layers):

        model.add(Dense(units=No_of_Units_in_dense_layers,activation=activation, kernel_regularizer=regularizers.l2(l2),

                  activity_regularizer=regularizers.l1(l1)))

    

    

    model.add(Dense(units=40,activation=activation))

    

    model.add(Dense(units=24,activation=activation))

    if optimizer=="Adam":

        opt=keras.optimizers.Adam(lr=lr, decay=decay, beta_1=0.9, beta_2=0.999)

    if optimizer=="Adagrad":

        opt=keras.optimizers.Adagrad(lr=lr, epsilon=None, decay=decay)

    if optimizer=="sgd":

        opt=keras.optimizers.SGD(lr=lr, momentum=momentum, decay=decay, nesterov=False)

    

    model.compile(loss=loss,optimizer=opt,

                 metrics=['accuracy'])

    

    out = model.fit(X_train, y_train, epochs=params['epochs'])



    return out,model
h = ta.Scan(x_train, y_train, params=params, model=Talos_Model, dataset_name='DR', experiment_no='1', grid_downsample=.01)
r = ta.Reporting(h)



r.best_params()