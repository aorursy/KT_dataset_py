import tensorflow as tf



import keras

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

from keras.layers import Dense, Activation, Dropout, Flatten



from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from keras.models import Sequential

from keras.layers import Dense , Activation , Dropout ,Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.metrics import categorical_accuracy

from keras.models import model_from_json

from keras.callbacks import ModelCheckpoint

from keras.optimizers import *

from keras.layers.normalization import BatchNormalization
# get the data

filname = '../input/facial-expression/fer2013/fer2013.csv'

label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

names=['emotion','pixels','usage']

df=pd.read_csv('../input/facial-expression/fer2013/fer2013.csv',names=names, na_filter=False)

im=df['pixels']

df.head(10)
df["usage"].value_counts()
train = df[["emotion", "pixels"]][df["usage"] == "Training"]

train.isnull().sum()
train['pixels'] = train['pixels'].apply(lambda im: np.fromstring(im, sep=' '))

x_train = np.vstack(train['pixels'].values)

y_train = np.array(train["emotion"])

x_train.shape, y_train.shape
public_test_df = df[["emotion", "pixels"]][df["usage"]=="PublicTest"]
public_test_df["pixels"] = public_test_df["pixels"].apply(lambda im: np.fromstring(im, sep=' '))

x_test = np.vstack(public_test_df["pixels"].values)

y_test = np.array(public_test_df["emotion"])
x_train = x_train.reshape(-1, 48, 48, 1)

x_test = x_test.reshape(-1, 48, 48, 1)

x_train.shape, x_test.shape
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)

y_train.shape, y_test.shape
import seaborn as sns

plt.figure(0, figsize=(12,6))

for i in range(1, 13):

    plt.subplot(3,4,i)

    plt.imshow(x_train[i, :, :, 0], cmap="gray")



plt.tight_layout()

plt.show()
def my_model():

    model = Sequential()

    input_shape = (48,48,1)

    model.add(Conv2D(64, (5, 5), input_shape=input_shape,activation='relu', padding='same'))

    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))

    model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))

    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Flatten())

    model.add(Dense(128))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(0.2))

    model.add(Dense(7))

    model.add(Activation('softmax'))

    

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')

    

    return model

model=my_model()

model.summary()
def my_model1():

    model = Sequential()

    input_shape = (48,48,1)

    model.add(Conv2D(64, (3,3), input_shape=input_shape,activation='relu', padding='same'))

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Conv2D(128, (3,3),activation='relu',padding='same'))

    model.add(Conv2D(128, (3,3),activation='relu',padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))

    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Flatten())

    model.add(Dense(128))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(0.2))

    model.add(Dense(7))

    model.add(Activation('softmax'))

    

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')

    

    return model

model1=my_model1()

model1.summary()
model1=my_model1()



hist = model1.fit(x_train, y_train, epochs=25,

                 shuffle=True,

                 batch_size=64, validation_data=(x_test, y_test), verbose=1)
model=my_model()



hist = model.fit(x_train, y_train, epochs=25,

                 shuffle=True,

                 batch_size=100, validation_data=(x_test, y_test), verbose=1)
test = df[["emotion", "pixels"]][df["usage"] == "PrivateTest"]

test["pixels"] = test["pixels"].apply(lambda im: np.fromstring(im, sep=' '))

test.head()
x_test_private = np.vstack(test["pixels"].values)

y_test_private = np.array(test["emotion"])
x_test_private = x_test_private.reshape(-1, 48, 48, 1)

y_test_private = np_utils.to_categorical(y_test_private)

x_test_private.shape, y_test_private.shape
score = model.evaluate(x_test_private, y_test_private, verbose=0)

score
score = model1.evaluate(x_test_private, y_test_private, verbose=0)

score