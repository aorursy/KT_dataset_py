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
train=pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')

test=pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')

sample_submission=pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv')
import matplotlib.pyplot as plt

%matplotlib inline
plt.figure(figsize=(20,10))

plt.plot(train[1:100000].time, train[1:100000].signal)

plt.show()
train.signal.value_counts()
X_train=train.drop(columns=['open_channels','time'])

y_train=train[['open_channels']]

test=test.drop(columns=['time'])
import tensorflow as tf

tf.__version__
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Activation, MaxPooling2D, Conv2D, Conv1D, MaxPooling1D, Convolution1D, Dropout

from keras.initializers import random_uniform

from tensorflow.keras.optimizers import SGD
X_train=X_train.values

y_train=y_train.values

test=test.values
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

test = test.reshape(test.shape[0], test.shape[1], 1)
X_train.shape

input_shape=X_train.shape[1:3]

input_shape
#hyperparameters

input_dimension = 226

learning_rate = 0.0025

momentum = 0.85

SEED = 42

hidden_initializer = random_uniform(seed=SEED)

dropout_rate = 0.2



model = Sequential()

model.add(Convolution1D(filters=16, kernel_size=1, input_shape=input_shape, activation='relu'))

model.add(Convolution1D(filters=16, kernel_size=1, activation='relu'))

model.add(Flatten())

model.add(Dropout(dropout_rate))

model.add(Dense(128, input_dim=input_dimension, kernel_initializer=hidden_initializer, activation='relu'))

model.add(Dropout(dropout_rate))

model.add(Dense(64, kernel_initializer=hidden_initializer, activation='relu'))

model.add(Dense(11, kernel_initializer=hidden_initializer, activation='softmax'))



sgd = SGD(lr=learning_rate, momentum=momentum)

model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['acc'])

model.fit(X_train, y_train, epochs=5, batch_size=128)
y_pred=model.predict_classes(test)
y_pred.shape
submission=pd.DataFrame({'time': sample_submission['time'], 'open_channels':y_pred})

submission.to_csv('/kaggle/working/submission.csv', float_format='%0.4f', index=False)

check=pd.read_csv('/kaggle/working/submission.csv')

check.head(15)
check.info()