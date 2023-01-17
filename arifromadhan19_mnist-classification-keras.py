# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



from keras.layers import Dense

from keras.datasets import mnist

from keras.layers import Dropout

from keras.utils import np_utils

from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
trainset = pd.read_csv('../input/train.csv')

testset = pd.read_csv('../input/test.csv')



print(trainset.shape)

print(testset.shape)



# split dataset into tainset and cross-validation set

trainset, valset= train_test_split(trainset, test_size = 0.1)



# get pix from trainset

x_train = trainset.loc[:, "pixel0" : "pixel783"]

x_train = x_train.values

# get label from trainset

y_train = trainset.loc[:,"label"]

y_train = pd.get_dummies(y_train).values



# get pix from valset

x_cv = valset.loc[:, "pixel0" : "pixel783"]

x_cv = x_cv.values

# get label from valset

y_cv = valset.loc[:,"label"]

y_cv = pd.get_dummies(y_cv).values



x_test = testset.values



print("Number of training examples: " + str(x_train.shape[1]))

print("Number of cross-validation examples = " + str(x_cv.shape[1]))

print ("x_train shape: " + str(x_train.shape))

print ("y_train shape: " + str(y_train.shape))

print ("y_train sample0: " + str(y_train[0]))

print ("x_test shape: " + str(x_test.shape))
# define baseline model

def network():

    # create model

    model = Sequential()

    model.add(Dense(256, input_dim=784, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer(),activation='relu'))

    model.add(Dense(128, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer(), activation='relu'))

    model.add(Dense(64, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer(), activation='relu'))

    model.add(Dense(32, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer(), activation='relu'))

    model.add(Dense(15, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer(), activation='relu'))

    model.add(Dense(10, kernel_initializer=tf.contrib.layers.xavier_initializer(), bias_initializer=tf.zeros_initializer(), activation='softmax'))

    # Compile model

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
# build the model

model = network()

# Fit the model

model.fit(x_train, y_train, validation_data=(x_cv, y_cv), epochs=30, batch_size=64, verbose=1)

# Final evaluation of the model

scores = model.evaluate(x_cv, y_cv, verbose=0)

print("Baseline Error: %.2f%%" % (100-scores[1]*100))

print("Baseline acc: ",scores[1]*100) 