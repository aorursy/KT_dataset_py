# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.import tensorflow as tf
import tensorflow as tf

data_dir = '../input/'

import os
print(os.listdir("../input"))

# load csv file
def load_data(row_nums):
    train = pd.read_csv(data_dir + 'train.csv').values
    x_test = pd.read_csv(data_dir + 'test.csv').values
    print(train.shape)

    x_train = train[:row_nums, 1:]
    y_train = train[:row_nums, 0]
    return x_train, y_train, x_test

x_train, y_train, x_test = load_data(42000)
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir='./logs', write_graph=True, write_images=False)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Data normalization from range(0,255) to range(0,1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, 10)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=[5, 5], input_shape=input_shape, strides=(1, 1), activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),
    tf.keras.layers.Conv2D(filters=64, kernel_size=[5, 5], activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1000, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

optimizer = tf.keras.optimizers.Adam(
    lr=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=None,
    decay=0.0,
    amsgrad=True)
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, callbacks=[tensorboard])
# score = model.evaluate(x_test, y_test, verbose=0)


### Export predictions to file for KAGGLE verification
y_pred = model.predict(x_test)

# output the prediction result
y_pred_max = y_pred.argmax(1)
pd.DataFrame({"ImageId": list(range(1,len(y_pred)+1)), "Label": y_pred_max}).to_csv('./Digit_Recogniser_Result.csv', index=False, header=True)
print(y_pred_max, len(y_pred_max))

