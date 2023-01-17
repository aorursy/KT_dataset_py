# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import struct

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

import keras

import tensorflow as tf

from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, BatchNormalization

from keras.utils import to_categorical

from keras.models import Sequential, Model

from keras.optimizers import Adam, Adadelta

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Input, BatchNormalization

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras import backend as K

from keras.datasets import mnist



print(K.image_data_format())
# the data, split between train and test sets

(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = to_categorical(y_train, num_classes = 10)

y_test = to_categorical(y_test, num_classes=10)
print("train samples", x_train.shape)

print("test samples", x_test.shape)
x_train = x_train.reshape(-1,28,28,1)

x_test = x_test.reshape(x_test.shape[0],28,28,1)
x_train.astype('float32')

x_test.astype('float32')
x_train = x_train/255

x_test = x_test/255
input_shape = (28,28,1)



model = Sequential()

model.add(Conv2D(96, (3, 3),

                 padding='Same',

                 activation='relu',

                 input_shape=input_shape))

model.add(BatchNormalization())

model.add(Conv2D(96,(3, 3),

                 padding='Same',

                 activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(192, (3,3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(Conv2D(256, (3,3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, (3,3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(Conv2D(256, (3,3), activation='relu', padding='same'))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.25))

model.add(Dense(units=10, activation='softmax'))
model.summary()
call_back = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=0, restore_best_weights=True)

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', 

                                              factor=0.25,

                                              verbose=1,

                                              patience=2,

                                              min_lr=0.000001)
model.compile(loss='categorical_crossentropy',

              optimizer=keras.optimizers.Adam(lr=0.001),

              metrics=['accuracy'])



train_datagen = ImageDataGenerator(shear_range=0.2,

                                   zoom_range=0.2,

                                   featurewise_center=False, # set input mean to 0 over the dataset

                                   samplewise_center=False,  # set each sample mean to 0

                                   featurewise_std_normalization=False,  # divide inputs by std of the dataset

                                   samplewise_std_normalization=False,  # divide each input by its std

                                   zca_whitening=False,  # apply ZCA whitening

                                   height_shift_range=.1,

                                   rotation_range=10,

                                   width_shift_range=.1)

train_datagen.fit(x_train)



history = model.fit_generator(

        train_datagen.flow(x_train,y_train, batch_size=128),

        steps_per_epoch=x_train.shape[0] // 128,

        epochs=32,

        validation_data=(x_test,y_test),callbacks=[call_back, reduce_lr])
print(history.history.keys())

#  "Accuracy"

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# "Loss"

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
xtest = pd.read_csv("../input/test.csv")
xtest = xtest.values.reshape(xtest.shape[0],28,28,1)

xtest = xtest / 255
submission = model.predict(xtest)



# Maximum Probability Index



submission = np.argmax(submission, axis = 1)

submission = pd.Series(submission,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),submission],axis = 1)



submission.to_csv("submission.csv",index=False)
submission.tail()