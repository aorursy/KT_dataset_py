%matplotlib inline

import pandas as pd

import numpy as np

import keras

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense,Conv2D, MaxPooling2D, Flatten

from sklearn.model_selection import train_test_split



# read data

train_data = pd.read_csv('../input/train.csv')
# split data into features and target label

features = train_data.iloc[:,1:]

target = np_utils.to_categorical(train_data.iloc[:,0])
# split data into train and test

x_train, x_test, y_train, y_test = train_test_split(features, target, train_size=0.7)
model = Sequential()

model.add(Dense(1024, activation='relu', input_dim=x_train.shape[1]))

model.add(Dense(728, activation='relu'))

model.add(Dense(512, activation='relu'))

model.add(Dense(256, activation='relu'))

model.add(Dense(y_train.shape[1], activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])



history = model.fit(np.asarray(x_train), np.asarray(y_train), batch_size=256, epochs=20, shuffle=True, verbose=0, validation_split=0.1)
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train_accuracy', 'validation_accuracy'], loc='best')

plt.show()