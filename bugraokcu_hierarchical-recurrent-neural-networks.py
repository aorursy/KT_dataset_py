# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from keras.utils import to_categorical

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split



data_train = pd.read_csv('../input/fashion-mnist_train.csv')

data_test = pd.read_csv('../input/fashion-mnist_test.csv')



img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)



X = np.array(data_train.iloc[:, 1:])

y = to_categorical(np.array(data_train.iloc[:, 0]))



#Here we split validation data to optimiza classifier during training

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)



#Test data

X_test = np.array(data_test.iloc[:, 1:])

y_test = to_categorical(np.array(data_test.iloc[:, 0]))







X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)



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

from keras.layers import LSTM



batch_size = 32

num_classes = 10

epochs = 2



row_hidden = 128

col_hidden = 128



row, col, pixel = X_train.shape[1:]



x = Input(shape=(row, col, pixel))



encoded_rows = TimeDistributed(LSTM(row_hidden))(x)



encoded_columns = LSTM(col_hidden)(encoded_rows)



prediction = Dense(num_classes, activation='softmax')(encoded_columns)

model = Model(x, prediction)

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



history = model.fit(X_train, y_train,

              batch_size=batch_size,

              epochs=epochs,

              verbose=1,

              validation_data=(X_val, y_val))

scores = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', scores[0])

print('Test accuracy:', scores[1])