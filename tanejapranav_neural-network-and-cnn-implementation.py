# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd

from keras.layers import Dense, Dropout, Activation, Reshape, Conv2D, AveragePooling2D, Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from keras.optimizers import adam

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
ds = pd.read_csv('../input/train.csv')
data = ds.values
split = int(0.8 * data.shape[0])

X_train = data[:split, 1:]/255.0
y_train = np_utils.to_categorical(data[:split, 0])

X_val = data[split:, 1:]/255.0
y_val = np_utils.to_categorical(data[split:, 0])

print (X_train.shape, y_train.shape)
print (X_val.shape, y_val.shape)
model = Sequential()
model.add(Reshape(target_shape=(1, 28, 28), input_shape = (784,)))

model.add(Conv2D(kernel_size = (5, 5), filters = 4, padding = "same"))

model.add(Conv2D(kernel_size = (5, 5), filters = 8, padding = "same"))
model.add(MaxPooling2D(pool_size = (2, 2), data_format="channels_first"))

model.add(Conv2D(kernel_size = (5, 5), filters = 16, padding = "same"))

model.add(Conv2D(kernel_size = (5, 5), filters = 32, padding = "same"))
model.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))

model.add(Conv2D(kernel_size = (5, 5), filters = 128, padding = "same"))

model.add(Flatten())

model.add(Dense(output_dim = 128, activation = 'relu'))

model.add(Dense(output_dim = 100, activation = 'relu'))
# Need an output of probabilities of all the 10 digits
model.add(Dense(output_dim = 10, activation= 'softmax'))
# adam = keras.optimizers.Adam(lr = 0.0005, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

X_train = data[:, 1:]/255.0
y_train = np_utils.to_categorical(data[:, 0])

model.fit(X_train, y_train, epochs = 30, batch_size=64)
test = pd.read_csv('../input/test.csv')
result = model.predict(test)
result = np.argmax(result,axis = 1)
result = pd.Series(result,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),result],axis = 1)
submission.to_csv("submission_ANN.csv",index=False)



