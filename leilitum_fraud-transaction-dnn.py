# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import tensorflow as tf

import keras



data = pd.read_csv('../input/creditcard.csv')
data.keys()
train = np.array(data.iloc[:,:30])
target = np.array(data.iloc[:,30])
train
model = keras.models.Sequential()
model.add(keras.layers.Dense(units=64, activation='relu', use_bias=True, input_shape=(30,)))
model.add(keras.layers.Dense(units=64, activation='relu', use_bias=True))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(units=32, activation='relu', use_bias=True))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1,activation='sigmoid'))
sgd = keras.optimizers.SGD(lr=0.001,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss=keras.losses.binary_crossentropy, optimizer=sgd, metrics=['binary_accuracy'])
model.fit(x=train,y=target, batch_size=32, shuffle=True, validation_split=0.1, epochs=5, class_weight={1:0.00172,0:0.99828})