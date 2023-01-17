# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

from tensorflow.keras.layers import Flatten, Dense

from keras.optimizers import RMSprop
print(tf.__version__)
x_train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')

y_train = x_train.label

x_train.drop(['label'], inplace = True, axis = 1)

x_test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.Sequential()

model.add(Dense(1024, activation = 'relu', input_dim = 784))

model.add(Dense(512, activation = 'relu'))

model.add((Dense(256, activation = 'relu')))

model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 50, verbose = 1, batch_size = 512)