# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd



df_train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

df_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers.core import Activation,Flatten,Dropout,Dense

from keras import backend as K

from keras.models import Sequential

from keras.optimizers import SGD
model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=( 28, 28,1), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), input_shape=( 28, 28,1), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax', name='predict'))
from keras.utils.np_utils import to_categorical

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import img_to_array

from keras.utils import to_categorical

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



import matplotlib.pyplot as plt

import numpy as np

import argparse

import random

import cv2

import os



trainY = df_train["label"]

trainX = df_train.drop(labels = ["label"],axis = 1) 

trainX = trainX / 255.0

df_test = df_test / 255.0

trainX = trainX.values.reshape(-1,28,28,1)

trainY = to_categorical(trainY, num_classes = 10)

random_seed = 42



trainX, X_val, trainY, Y_val = train_test_split(trainX, trainY, test_size = 0.2, random_state=random_seed)



from keras.optimizers import RMSprop

optimizer = RMSprop(lr=0.001)

# Compile the model

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
epochs = 10 

batch_size = 32



datagen = ImageDataGenerator(rotation_range=10, zoom_range = 0.1, width_shift_range=0.1, height_shift_range=0.1,horizontal_flip=False,   vertical_flip=False)  

datagen.fit(trainX)

history = model.fit_generator(datagen.flow(trainX,trainY, batch_size=batch_size),

epochs = epochs, validation_data = (X_val,Y_val), verbose = 2, steps_per_epoch=trainX.shape[0] // batch_size)

df_test.shape
df_test = df_test.values.reshape(-1,28,28,1)

results = model.predict(df_test)



results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)


