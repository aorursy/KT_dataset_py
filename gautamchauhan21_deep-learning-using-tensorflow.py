import numpy as np 

import tensorflow as tf

import pandas as pd

import os



print(os.listdir("../input"))

train_data = pd.read_csv('../input/fashion-mnist_train.csv')



test_data = pd.read_csv('../input/fashion-mnist_test.csv')
train_data.sample(10)
x_train = train_data.drop('label',axis=1)

x_test = test_data.drop('label',axis=1)

y_train = train_data.label

y_test = test_data.label



x_train, x_test = x_train/255.0,x_test/255.0
model = tf.keras.models.Sequential([

    tf.keras.layers.Dense(512, activation = tf.nn.relu),

    tf.keras.layers.Dense(10, activation = tf.nn.softmax)

])



model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])





x_tr = x_train.values

y_tr = y_train.values
model.fit(x_tr, y_tr, epochs=10)
x_te = x_test.values

y_te = y_test.values



model.evaluate(x_te, y_te)