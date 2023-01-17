import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow import keras
data_train = pd.read_csv('../input/fashion-mnist_train.csv')

data_test = pd.read_csv('../input/fashion-mnist_test.csv')



X_train = np.array(data_train.iloc[:, 1:])

X_test = np.array(data_test.iloc[:, 1:])



Y_train = keras.utils.to_categorical(np.array(data_train.iloc[:, 0]))

Y_test = keras.utils.to_categorical(np.array(data_test.iloc[:, 0]))

img_rows, img_cols = 28, 28

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)



X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train /= 255

X_test /= 255
from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D



input_shape = (img_rows, img_cols, 1)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(250, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',

          loss='categorical_crossentropy',

          metrics=['accuracy'])
model.fit(X_train, Y_train,

          batch_size=50,

          epochs=10,

          validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test)
print('Accuracy: ',score[1])