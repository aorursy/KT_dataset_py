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
from tensorflow import keras
train_data = pd.read_csv("../input/digit-recognizer/train.csv")
train_data
features = train_data.drop(['label'], axis=1)
labels = train_data['label']

train_data.shape

model1 = keras.Sequential()
model1.add(keras.layers.Dense(300,input_shape=(pixels,),activation=tf.nn.relu))
model1.add(keras.layers.Dense(10,activation=tf.nn.softmax))
model1.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model1.fit(x=features, y=labels, batch_size=64, epochs=10)
model2 = keras.Sequential()
model2.add(keras.layers.Dense(300,input_shape=(pixels,),activation=tf.nn.relu))
model2.add(keras.layers.Dense(10,activation=tf.nn.softmax))
model2.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model2.fit(x=features, y=labels, batch_size=64, epochs=10)
model3 = keras.Sequential()
model3.add(keras.layers.Dense(300,input_shape=(pixels,),activation=tf.nn.relu))
model3.add(keras.layers.Dense(10,activation=tf.nn.softmax))
model3.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model3.fit(x=features, y=labels, batch_size=64, epochs=10)
model4 = keras.Sequential()
model4.add(keras.layers.Dense(300,input_shape=(pixels,),activation=tf.nn.relu))
model4.add(keras.layers.Dense(10,activation=tf.nn.softmax))
model4.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model4.fit(x=features, y=labels, batch_size=64, epochs=10)
features_mean = features-sum(features.sum())/(features.shape[0]*features.shape[1])
features
features_mean
n_model1 = keras.Sequential()
n_model1.add(keras.layers.Dense(261,input_shape=(pixels,),activation=tf.nn.relu))
n_model1.add(keras.layers.Dense(10,activation=tf.nn.softmax))
n_model1.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
n_model1.fit(x=features_mean, y=labels, batch_size=64, epochs=20)
n_model2 = keras.Sequential()
n_model2.add(keras.layers.Dense(210,input_shape=(pixels,),activation=tf.nn.relu))
n_model2.add(keras.layers.Dense(10,activation=tf.nn.softmax))
n_model2.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
n_model2.fit(x=features_mean, y=labels, batch_size=64, epochs=20)
n_model3 = keras.Sequential()
n_model3.add(keras.layers.Dense(300,input_shape=(pixels,),activation=tf.nn.relu))
n_model3.add(keras.layers.Dense(10,activation=tf.nn.softmax))
n_model3.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
n_model3.fit(x=features_mean, y=labels, batch_size=64, epochs=20)
n_model4 = keras.Sequential()
n_model4.add(keras.layers.Dense(350,input_shape=(pixels,),activation=tf.nn.relu))
n_model4.add(keras.layers.Dense(10,activation=tf.nn.softmax))
n_model4.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
n_model4.fit(x=features_mean, y=labels, batch_size=64, epochs=20)
n_model4 = keras.Sequential()
n_model4.add(keras.layers.Dense(400,input_shape=(pixels,),activation=tf.nn.relu))

n_model4.add(keras.layers.Dense(10,activation=tf.nn.softmax))
n_model4.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
n_model4.fit(x=features_mean, y=labels, batch_size=64, epochs=20)
n_model4.a