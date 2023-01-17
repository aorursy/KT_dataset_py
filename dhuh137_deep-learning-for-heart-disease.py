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
from sklearn.model_selection import train_test_split

df = pd.read_csv('../input/heart.csv')

X = df.drop(['target'],axis=1)

Y = df['target']

x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle = True, test_size=0.30)
import keras as K
from keras.utils import to_categorical

y_binary = to_categorical(y_train)

model = K.models.Sequential(

          [K.layers.Dense(256, input_shape = (x_train.shape[1],), activation = "relu", kernel_initializer='random_uniform',

                bias_initializer='ones'),

           K.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),

           K.layers.Dense(128, activation="relu", kernel_initializer='random_uniform',

                bias_initializer='ones'),

           K.layers.Dropout(0.5),

           K.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),

           K.layers.Dense(64, activation="relu", kernel_initializer='random_uniform',

                bias_initializer='ones'),

           K.layers.Dropout(0.5),

           K.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),

           K.layers.Dense(32, activation="relu", kernel_initializer='random_uniform',

                bias_initializer='ones'),

           K.layers.Dense(y_binary.shape[1], activation = "softmax"),

          ])

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])

tbCallBack = K.callbacks.TensorBoard(log_dir='./Graph/', histogram_freq=0, write_graph=True, write_images=True)

model.fit(x_train,y_binary, epochs=100, batch_size =128, callbacks=[tbCallBack])

model.save('./model-v1.h5')
y_binary_test = to_categorical(y_test)

print(model.evaluate(x_test,y_binary_test))

print(model.evaluate(x_train,y_binary))