!pip install tensorflow==2.0.0-alpha0
#%load_ext tensorboard.notebook

#%tensorboard --logdir logs
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.preprocessing import Normalizer

#!pip install tensorflow==2.0.0-alpha0
print(tf.__version__)
#df = pd.read_csv('../input/housing.csv')
#df.shape
data = tf.keras.datasets.boston_housing
(x_train, y_train),(x_test, y_test) = data.load_data()
x_train = tf.keras.utils.normalize(x_train)

#y_train = tf.keras.utils.normalize(y_train)

x_test = tf.keras.utils.normalize(x_test)

#y_test = tf.keras.utils.normalize(y_test)
x_train.shape, x_test.shape, y_train.shape, y_test.shape
type(x_train)
model = tf.keras.models.Sequential([

  tf.keras.layers.Flatten(input_shape=(13, )),

  tf.keras.layers.Dense(23, activation=tf.nn.relu),

  tf.keras.layers.Dense(23, activation=tf.nn.relu),

  tf.keras.layers.Dense(23, activation=tf.nn.relu),

  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(1)

])
type(model.weights), len(model.weights), model.weights[0].shape
model.weights[0][0]
model.compile(optimizer='adam',

              loss='mean_squared_error',

              metrics=['mean_squared_error'])
history = model.fit(x_train, y_train, 

                    epochs=500, 

                    verbose = 0)



model.evaluate(x_test, y_test)
history.history.keys()
fig, axes = plt.subplots(figsize = (12,4))

plt.subplot(1, 2, 1)

plt.plot(history.history['loss'])

plt.xlabel('epoch')

plt.ylabel('loss')

plt.subplot(1, 2, 2)

plt.plot(history.history['mean_squared_error'])

plt.xlabel('epoch')

plt.ylabel('mean_squared_error')

None