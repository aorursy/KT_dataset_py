from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image, ImageFile

from IPython.display import display

print(os.listdir('../input'))
images = np.load('../input/images.npy')

print(images.shape)

labels = np.load('../input/labels.npy')

print(labels)
im = Image.fromarray(images[1000])

display(im)
nRowsRead = 1000 # specify 'None' if want to read whole file

# labels.csv has 2000 rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('../input/labels.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'labels.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
from keras.models import Sequential



model = tf.keras.models.Sequential([

  tf.keras.layers.Conv2D(64, (3, 3), padding="same",input_shape=(480, 640, 3)),

  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

  tf.keras.layers.Conv2D(32, (3, 3), padding="same"),

  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

  tf.keras.layers.Conv2D(16, (3, 3), padding="same"),

  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

  tf.keras.layers.Conv2D(8, (3, 3), padding="same"),

  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

  tf.keras.layers.Flatten(),

  tf.keras.layers.Dense(512, activation='relu'),

  tf.keras.layers.Dropout(0.5),

  tf.keras.layers.Dense(128, activation='relu'),

  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(1),

])



model.compile(optimizer='adam',

              loss='mean_squared_error',

              metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'])



model.fit(images, labels, epochs=30, batch_size = 32, validation_split = 0.2, shuffle=True)
model.predict(images[0:10])

labels[0:10]