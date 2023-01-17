# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import keras

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Flatten

from keras.optimizers import SGD, Adam, RMSprop

from keras.utils import np_utils

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras import backend as K

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1) 

train.shape
test.shape
X_train = X_train.reshape(-1,28,28,1)

test = test.reshape(-1,28,28,1)
training_images,test_images, training_labels, test_labels=train_test_split(X_train , Y_train, test_size=0.33, random_state=4)

training_images.shape
test_images.shape






training_images=training_images.reshape(-1, 28, 28, 1)

training_images=training_images / 255.0

test_images = test_images.reshape(-1, 28, 28, 1)

test_images=test_images/255.0

##Model 
import tensorflow as tf

model = tf.keras.models.Sequential([

  tf.keras.layers.Conv2D(64, (5,5), activation='relu', input_shape=(28, 28, 1)),

  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Conv2D(64, (5,5), activation='relu'),

  tf.keras.layers.MaxPooling2D(2,2),

  tf.keras.layers.Flatten(),

  tf.keras.layers.Dense(128, activation='relu'),

  tf.keras.layers.Dense(10, activation='softmax')

])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(training_images, training_labels, epochs=5)

test_loss = model.evaluate(test_images, test_labels)