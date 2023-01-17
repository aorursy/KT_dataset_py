import numpy as np

import pandas as pd

import os

import cv2

import keras

import matplotlib.pyplot as plt

import sklearn

import seaborn as sns

from PIL import Image
from keras.models import Sequential, Model

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Dense, Dropout, Lambda, Activation, Flatten, Input

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam, RMSprop, SGD

from keras.utils import np_utils
from sklearn.model_selection import train_test_split
!ls /kaggle/input/behavioural-cloning/IMG/IMG/ | wc -l
df = pd.read_csv('/kaggle/input/behavioural-cloning/driving_log.csv', names=['center', 'left', 'right', 'streeing_angle', 'speed', 'throttle', 'brake'])

df.head()
df.count()
def clean_img_path(imgs):

    imgs = imgs.split('\\')[-1]

    

    return imgs
df['center'] = df['center'].apply(clean_img_path)

df.head()
df['left'] = df['left'].apply(clean_img_path)

df.head()
df['right'] = df['right'].apply(clean_img_path)

df.head()
# Model

model = keras.models.Sequential()

model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(66, 200, 3)))

model.add(Conv2D(24, (5, 5), activation='relu', strides=(2, 2)))

model.add(Conv2D(36, (5, 5), activation='relu', strides=(2, 2)))

model.add(Conv2D(48, (5, 5), activation='relu', strides=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Flatten())

model.add(Dense(100, activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(10, activation='relu'))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.01))

model.summary()
df = sklearn.utils.shuffle(df)

df.head(10)
X_train, X_valid, y_train, y_valid = train_test_split(df[['center', 'left', 'right']], df['streeing_angle'], test_size=0.10, random_state=2020)
X_train
y_train
X_valid
y_valid
def img_batch(imgs, labels):

    while True:

        for i, label in zip(imgs, labels):

            # print(np.random.permutation(i)[0], label)

            img = cv2.imread('/kaggle/input/behavioural-cloning/IMG/IMG/' + np.random.permutation(i)[0])

            img = cv2.resize(img, (200, 66))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            

            yield img, label



img_batch_trn_gen = img_batch(X_train.values, y_train)

img_batch_val_gen = img_batch(X_valid.values, y_valid)
temp, label = next(img_batch_trn_gen)

temp.shape
temp, label = next(img_batch_val_gen)

temp.shape
def batch_for_network_generator(gen, batch_size):

    while True:

        # batch_of_imgs = [next(img_batch_gen) for i in range(64)]

        x = np.ones((batch_size, 66, 200, 3), dtype=np.uint8)

        y = np.zeros((64))

        for i in range(0, batch_size):

            x[i], y[i] = next(gen)

        

        yield (x, y)
t_x, t_y = next(batch_for_network_generator(img_batch_trn_gen, 64))
t_x.shape
t_y.shape
X_train.shape, X_valid.shape
model.fit_generator(batch_for_network_generator(img_batch_trn_gen, 64), steps_per_epoch=2229//64, epochs=2, verbose=1, 

                    validation_data=batch_for_network_generator(img_batch_val_gen, 64), validation_steps=248//64)