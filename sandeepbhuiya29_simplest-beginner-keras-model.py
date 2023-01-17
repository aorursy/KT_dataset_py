import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

import seaborn as sns

import zipfile

from tqdm import tqdm

import os
local_zip = "../input/facial-keypoints-detection/training.zip"

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('/tmp/train')

zip_ref.close()
local_zip = "../input/facial-keypoints-detection/test.zip"

zip_ref = zipfile.ZipFile(local_zip, 'r')

zip_ref.extractall('/tmp/test')

zip_ref.close()
training_directory = "/tmp/train/"

testing_directory = "/tmp/test/"



print(os.listdir(training_directory))

print(os.listdir(testing_directory))
train = pd.read_csv("/tmp/train/training.csv")

test = pd.read_csv("/tmp/test/test.csv")

train.head(5)
train.info()
train.isnull().any().value_counts()
train.fillna(method = 'ffill',inplace = True)
X = train.Image.values

del train['Image']

Y = train.values
x = []

for i in tqdm(X):

    q = [int(j) for j in i.split()]

    x.append(q)

len(x)
x = np.array(x)

x = x.reshape(7049, 96,96,1)

x  = x/255.0

x.shape
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,Y,random_state = 42,test_size = 0.3)
x_train.shape,x_test.shape
y_train.shape,y_test.shape
from keras.layers.advanced_activations import LeakyReLU

from keras.models import Sequential, Model

from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from keras.optimizers import Adam
model = Sequential()



model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(96,96,1)))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))

model.add(BatchNormalization())

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())





model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(30))

model.summary()
model.compile(optimizer = 'adam',loss = 'mean_squared_error', metrics = ['mae','acc'])

model.fit(x_train,y_train,batch_size=256, epochs=100,validation_data=(x_test,y_test))
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])