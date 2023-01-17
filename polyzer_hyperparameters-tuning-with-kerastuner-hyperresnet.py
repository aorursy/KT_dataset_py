!pip install -U keras-tuner
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from __future__ import print_function

import keras

import tensorflow as tf

from keras.datasets import mnist

from keras import backend as K

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from kerastuner.applications import HyperResNet

from kerastuner.tuners import Hyperband



#our training parameters

batch_size = 500

num_classes = 10

epochs = 15



# input image dimensions

img_rows, img_cols = 28, 28

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
test_df = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

train_df = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
Y_train_df = train_df['label']

#Y_train_df = pd.get_dummies(Y_train_df).values

#Y_train_df = Y_train_df.values.reshape(-1, 1)

#Y_train_df = standard_scaler.fit_transform(Y_train_df)

#Y_train_df = keras.utils.np_utils.to_categorical(Y_train_df)

#Y_train_df = Y_train_df.reshape(-1, 10, 1)

print(Y_train_df.shape)

X_train_df = train_df.drop(columns="label")

#X_train_df = standard_scaler.fit_transform(X_train_df)

#X_train_df = keras.utils.np_utils.to_categorical(X_train_df)

print(X_train_df.shape)
X_test_df = test_df

#X_train_df = standard_scaler.fit_transform(X_train_df)

#X_train_df = keras.utils.np_utils.to_categorical(X_train_df)

print(X_test_df.shape)
np.random.seed(7)

# split into 67% for train and 15% for test

X_train, X_valid, y_train, y_valid = train_test_split(X_train_df, Y_train_df, test_size=0.15, random_state=7)
# Artificially increase training set

train_datagen = ImageDataGenerator(rescale=1./255.,

                                   rotation_range=10,

                                   width_shift_range=0.25,

                                   height_shift_range=0.25,

                                   shear_range=0.1,

                                   zoom_range=0.25,

                                   horizontal_flip=False)



valid_datagen = ImageDataGenerator(rescale=1./255.)
X_test = test_df.drop('id', axis = 1)

# X_test = standard_scaler.fit_transform(X_test)

y_test = test_df['id']

# y_test = keras.utils.np_utils.to_categorical(y_test)

y_test
x_train = X_train.values

x_valid = X_valid.values

y_train = y_train.values

y_valid = y_valid.values

x_test = X_test.values
if K.image_data_format() == 'channels_first':

    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)

    x_valid = x_valid.reshape(x_valid.shape[0], 1, img_rows, img_cols)

    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)

    input_shape = (1, img_rows, img_cols)

else:

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

    x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols, 1)

    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')

x_valid = x_valid.astype('float32')

x_test = x_test.astype('float32')

# x_train /= 255

# x_valid /= 255

# x_test /= 255

print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_valid.shape[0], 'test samples')
# convert class vectors to binary class matrices

y_train = keras.utils.to_categorical(y_train, num_classes)

y_valid = keras.utils.to_categorical(y_valid, num_classes)
def get_tuning_model(num_classes, input_shape):

    model = HyperResNet(input_shape=input_shape, classes=num_classes)

    return model
model = get_tuning_model(num_classes, input_shape)

tuner = Hyperband(

    model,

    objective='val_accuracy',

    max_epochs=2,

    directory='my_dir',

    project_name='helloworld')



tuner.search(x_train, y_train, epochs=2, validation_data=(x_valid, y_valid))