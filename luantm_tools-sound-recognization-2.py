import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import tensorflow as tf

import pickle

import librosa

import IPython.display as ipd
classes = os.listdir("../input/tools-sound/train")

classes_vn = [u'cưa máy', u'đao kiếm', u'còi', u'máy khoan', u'búa']

print(classes)
def create_data():

    x = []

    y = []

    for c in classes:

#         try:

            tmpx = []

            tmpy = []

            print('processs-...', c)



            for file in os.listdir('../input/tools-sound/train/' + c):

                wave,sr = librosa.load('../input/tools-sound/train/' + c +'/' + file, mono=True)

                mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=20)

                mfcc_pad = np.zeros((20, 44))

                mfcc_pad[:mfcc.shape[0], :mfcc.shape[1]] = mfcc[:20, :44]

                print(file)

                if mfcc_pad.shape == (20, 44):

                    x.append(mfcc_pad)

                    tmpx.append(mfcc_pad)

                    y.append(classes.index(c))

                    tmpy.append(classes.index(c))



            print('write file pickle ', c)

#             pickle.dump(np.array(tmpx), open('{}.pickle'.format(c), 'wb'))

#             pickle.dump(np.array(tmpy), open('{}_y.pickle'.format(c), 'wb'))

#         except:

#             pass

            



    print('complete')

    return np.array(x), np.array(y)

        
x, y = create_data()
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from sklearn.utils import shuffle

x, y = shuffle(x, y)

x_train = x.reshape(-1, 20, 44, 1)

y_train = y.reshape(-1)

y_train = to_categorical(y, num_classes = 5)



print(x_train.shape)

print(y_train.shape)
# print(x.shape)

print(x)
from keras import regularizers

from keras.datasets import cifar10

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.optimizers import SGD

from keras.callbacks import ModelCheckpoint

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization

from keras import regularizers

from keras.callbacks import LearningRateScheduler

from keras import backend as K

import keras

import matplotlib.pyplot as plt

weight_decay = 1e-4

model = Sequential()

model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))

model.add(Activation('relu'))

model.add(BatchNormalization())

# model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

# model.add(Activation('relu'))

# model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))

model.add(Activation('relu'))

model.add(BatchNormalization())

# model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

# model.add(Activation('relu'))

# model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))

model.add(Activation('relu'))

model.add(BatchNormalization())

# model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

# model.add(Activation('relu'))

# model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))

model.add(Activation('relu'))

model.add(BatchNormalization())

# model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

# model.add(Activation('relu'))

# model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))

 

# model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

# model.add(Activation('relu'))

# model.add(BatchNormalization())

# # model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

# # model.add(Activation('relu'))

# # model.add(BatchNormalization())

# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Dropout(0.3))

 

# model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

# model.add(Activation('relu'))

# model.add(BatchNormalization())

# model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))

# model.add(Activation('relu'))

# model.add(BatchNormalization())

# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Dropout(0.4))

 

model.add(Flatten())

model.add(Dense(5, activation='softmax'))

opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
model.fit(x=x_train,

          y=y_train, epochs=100, validation_split=0.1)
test = os.listdir('../input/tools-sound/test')

# for i in test:

#     print(i)

print('../input/tools-sound/test/' + test[6])
ipd.Audio('../input/tools-sound/test/' + test[10]) # load a local WAV file
wave,sr = librosa.load('../input/tools-sound/test/' + test[10], mono=True)

mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=20)

mfcc_pad = np.zeros((20, 44))

mfcc_pad[:mfcc.shape[0], :mfcc.shape[1]] = mfcc[:20, :44]

x_pred = mfcc_pad.reshape(-1, 20, 44, 1)

result_index = np.argmax(model.predict(x_pred))

print(result_index)

print('Dự đoán đây là tiếng: ' + classes_vn[result_index])