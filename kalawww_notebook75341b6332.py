import os

import pandas

import numpy as np



from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.layers.core import Dense, Flatten, Dropout

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator



import keras
TRAIN_FILENAME = '../input/train.csv'

TEST_FILENAME = '../input/test.csv'
img_shape = (1, 28, 28)

X_train = train[:, 1:].reshape(train.shape[0], 1, 28, 28)

Y_train = keras.utils.np_utils.to_categorical(train[:, 0], 10)

X_test = test.reshape(test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')

X_train /= 255



X_test = X_test.astype('float32')

X_test /= 255
split_size = int(X_train.shape[0] * 0.9)

X_train, X_valid = X_train[:split_size], X_train[split_size:]

Y_train, Y_valid = Y_train[:split_size], Y_train[split_size:]
datagen_train = ImageDataGenerator(

    rotation_range=20,

    width_shift_range=0.15,

    height_shift_range=0.15

)

datagen_valid = ImageDataGenerator(

    rotation_range=20,

    width_shift_range=0.15,

    height_shift_range=0.15

)
datagen_train.fit(X_train)

datagen_valid.fit(X_valid)
model = Sequential()



model.add(BatchNormalization(axis=1, input_shape=img_shape))



model.add(Convolution2D(32,5,5, activation='relu'))

model.add(MaxPooling2D((2,2)))



model.add(Convolution2D(64,5,5, activation='relu'))

model.add(MaxPooling2D((2,2)))



model.add(Convolution2D(128,4,4, activation='relu'))



model.add(Flatten())

model.add(Dense(200, activation='relu'))

model.add(BatchNormalization(axis=1))

model.add(Dropout(0.5))

model.add(Dense(100, activation='relu'))

model.add(BatchNormalization(axis=1))

model.add(Dropout(0.5))

model.add(Dense(100, activation='relu'))

model.add(BatchNormalization(axis=1))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
n_epoch = 100

val_split = 0.1

batch_size = 16
model.compile(

    "adamax",

    loss='categorical_crossentropy',

    metrics=['accuracy']

)
np.savetxt(

    '../input/pred.csv',

    np.c_[range(1, len(preds)+1), preds],

    delimiter=',',

    header='ImageId,Label',

    comments='',

    fmt='%d'

)