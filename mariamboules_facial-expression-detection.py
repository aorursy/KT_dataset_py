# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

# !pip install utils

# import utils



import csv

from PIL import Image    

from sklearn.model_selection import train_test_split

from tqdm import tqdm

import numpy as np # linear algebra

from sklearn.datasets import make_classification

from imblearn.over_sampling import RandomOverSampler

import collections

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf

from keras.preprocessing import image

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import ImageDataGenerator

from keras.layers.normalization import BatchNormalization

from tensorflow.keras import datasets, layers, models

from keras.layers import Dense, Activation

from tensorflow.keras.layers import Dropout

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# from tensorflow.keras.utils import plot_model

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        print(os.listdir("../input"))

        



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/train.csv')

test = pd.read_csv('/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/test.csv')

icml_face_data = pd.read_csv('/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/icml_face_data.csv')


oversample = RandomOverSampler(sampling_strategy='auto')

# fit and apply the transform

X_over, y_over = oversample.fit_resample((train.pixels).values.reshape(-1, 1), train.emotion)

a = np.array( y_over)

collections.Counter(a)

y_over = pd.Series(y_over)

y_over= y_over.values.reshape(len(y_over),1)
a= []

X_over = pd.Series(X_over.flatten())

for i in range(len(X_over)):

        image_string = (X_over)[i].split(' ') 

        image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48,1)

        a.append(image_data)



X_train = np.array(a)

X_test =test

Y_train = y_over

print(a)

print ("number of training examples = " + str(X_train.shape[0]))

print ("number of test examples = " + str(X_test.shape[0]))

print ("X_train shape: " + str(X_train.shape))

print ("Y_train shape: " + str(Y_train.shape))

#print ("X_test shape: " + str(X_test.shape))


model = models.Sequential()

model.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))

model.add(BatchNormalization())

model.add(layers.MaxPooling2D((2, 2)))

model.add(Dropout(0.25))



model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(layers.MaxPooling2D((2, 2)))

model.add(Dropout(0.25))



model.add(layers.Conv2D(256, (5, 5), activation='relu'))

model.add(BatchNormalization())

model.add(layers.MaxPooling2D((2, 2)))

model.add(Dropout(0.25))



model.add(layers.Flatten())



model.add(layers.Dense(128))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.25))



model.add(layers.Dense(256))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.25))



model.add(layers.Dense(7, activation='softmax'))

model.summary()



model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])



model.fit(X_train, Y_train, batch_size=64, epochs=10, steps_per_epoch=(len(X_train)/128))



model.save('CNNmodel')
#print(X_test.pixels[0])

a= []

for i in range(len(X_test.pixels)):

        image_string = (X_test.pixels)[i].split(' ') 

        image_data = np.asarray(image_string, dtype=np.uint8).reshape(48,48,1)

        a.append(image_data)

#print(a)        



X_testing = np.array(a)

prediction = model.predict(X_testing)

print(prediction)