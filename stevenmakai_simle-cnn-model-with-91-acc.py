import numpy as np 

import pandas as pd 

import tensorflow as tf

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation

train = pd.read_csv("../input/fashion-mnist_train.csv")

test = pd.read_csv("../input/fashion-mnist_test.csv")
train.columns
y_train = train['label'].values

x_train = train.drop('label', axis  = 1).values

y_test = test['label'].values

x_test = test.drop('label', axis  = 1).values

x_train = x_train.reshape((-1,28,28,1))

x_test = x_test.reshape((-1,28,28,1))

x_train.shape, y_train.shape, x_test.shape, y_test.shape
model = tf.keras.models.Sequential([

    Conv2D(16, kernel_size=(3,3), input_shape=(28,28,1)),

    BatchNormalization(),

    Activation('relu'),

    Conv2D(16, (3)),

    BatchNormalization(),

    Activation('relu'),

    MaxPooling2D(pool_size=(2,2)),

    Dropout(0.5),

    Conv2D(32, kernel_size=(3,3)),

    BatchNormalization(),

    Activation('relu'),

    Conv2D(32, (3,3)),

    BatchNormalization(),

    Activation('relu'),

    MaxPooling2D(pool_size=(2,2)),

    Dropout(0.5),

    Flatten(),

    Dense(128, activation='relu'),

    Dropout(0.5),

    Dense(10, activation='softmax')

])



model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])





model.fit(x_train, y_train, epochs=100, validation_split = 0.2, batch_size = 1000)



model.evaluate(x_test, y_test)