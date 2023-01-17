import numpy as np

import pandas as pd

import os

print(os.listdir("../input")) # data files are available in the "../input/" directory.
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

import matplotlib.pyplot as plt

from keras.utils import plot_model



# input image dimensions

img_rows, img_cols = 28, 28



# read data

train = pd.read_csv('../input/train.csv')



x_train = train.iloc[:,1:].values.astype('float32')

y_train = train.iloc[:,0].values.astype('int32')



# reshape to (train_size, 28, 28, 1)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)



x_test = x_train[0:8000,:]

y_test = y_train[0:8000]

trainset_size = 10000  # max 36000

x_train = x_train[-trainset_size:,:]

y_train = y_train[-trainset_size:]

y_train = keras.utils.to_categorical(y_train)  # one-hot encode the output, only for the training set



print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')
# build model

model = Sequential()

model.add(Conv2D(32, kernel_size=(8, 8), activation='relu', input_shape=input_shape))

model.add(Conv2D(64, (8, 8), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))  # reducing the size of the feature maps

model.add(Dropout(0.25))  # regularization by ignoring randomly selected neurons during training 

model.add(Flatten())  # from 2-dimensional back to 1-dimensional

model.add(Dense(64, activation='relu'))  # a "normal" layer

model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))  # output layer

# note that dropout and flatten are not really layers with parameters that can be trained



earlystopping=[EarlyStopping(monitor='loss', patience=5, verbose=1, mode='auto')]



plot_model(model, to_file='model.png')
model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),  # optimized gradient descent with automatically updated learning rate

              metrics=['accuracy'])



history = model.fit(x_train, y_train,

          batch_size=128,  # mini batch size

          epochs=60,  # is the maximum #epochs

          verbose=1,

          validation_split=0.2,

          callbacks=earlystopping)
# accuracy as a function of #epochs

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



# loss as a function of #epochs

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()