import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from matplotlib import pyplot as plt

from PIL import Image
dftrain = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

dftest = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

X_train = dftrain.drop(columns='label').values.reshape((dftrain.shape[0], 28, 28, 1))

y_train = keras.utils.to_categorical(dftrain['label'].values)

X_test = dftest.values.reshape((dftest.shape[0], 28, 28, 1))
x = tf.image.grayscale_to_rgb(tf.constant(X_train))

x = tf.pad(x, ((0, 0), (2, 2), (2, 2), (0, 0)))

x = keras.applications.resnet_v2.preprocess_input(tf.cast(x, tf.float32))
lambda_ = 1e-3

model = keras.Sequential()

core = keras.applications.ResNet152V2(

    include_top=False,

    weights='imagenet',

    input_shape=x.shape[1:]

)

for layer in core.layers[:-2]:

    layer.trainable = False

model.add(core)

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(10, activation='softmax', 

                             kernel_regularizer=keras.regularizers.l2(lambda_)))
model.summary()
model.compile(

    optimizer=keras.optimizers.Adam(),

    loss='categorical_crossentropy',

    metrics=['accuracy']

)
history = model.fit(x, y_train, validation_split=0.2, batch_size=128, epochs=20)