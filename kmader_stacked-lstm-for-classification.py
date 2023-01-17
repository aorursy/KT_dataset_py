# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.utils import to_categorical

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split



data_train = pd.read_csv('../input/fashion-mnist_train.csv')

data_test = pd.read_csv('../input/fashion-mnist_test.csv')



img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols)



X = np.array(data_train.iloc[:, 1:])

y = to_categorical(np.array(data_train.iloc[:, 0]))



#Here we split validation data to optimiza classifier during training

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)



#Test data

X_test = np.array(data_test.iloc[:, 1:])

y_test = to_categorical(np.array(data_test.iloc[:, 0]))







X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols)

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols)

X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols)



X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_val = X_val.astype('float32')

X_train /= 255

X_test /= 255

X_val /= 255
import keras

from keras.datasets import mnist

from keras.models import Model

from keras.layers import Input, Dense, TimeDistributed

from keras.layers import LSTM, Bidirectional, Conv1D, concatenate, Permute, Dropout



batch_size = 32

num_classes = 10

epochs = 2



row_hidden = 128

col_hidden = 128



row, col = X_train.shape[1:]



input = Input(shape=(row, col))



def lstm_pipe(in_layer):

    x = Conv1D(row_hidden, kernel_size=3, padding = 'same')(in_layer)

    x = Conv1D(row_hidden, kernel_size=3, padding = 'same')(x)

    encoded_rows = Bidirectional(LSTM(row_hidden, return_sequences = True))(x)

    return LSTM(col_hidden)(encoded_rows)

# read it by rows

first_read = lstm_pipe(input)

# read it by columns

trans_read = lstm_pipe(Permute(dims = (1,2))(input))

encoded_columns = concatenate([first_read, trans_read])

encoded_columns = Dropout(0.2)(encoded_columns)

prediction = Dense(num_classes, activation='softmax')(encoded_columns)

model = Model(input, prediction)

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

model.summary()
history = model.fit(X_train, y_train,

              batch_size=batch_size,

              epochs=epochs,

              verbose=1,

              validation_data=(X_val, y_val))
scores = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', scores[0])

print('Test accuracy:', scores[1])