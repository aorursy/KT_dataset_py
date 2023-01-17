import numpy as np

from numpy import array

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import ShuffleSplit

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
mnist = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
mnist.head()
#hyperparameters

LABELS = 10           # Number of labls(1-10)

WIDTH = 28      # Width/height if the image

COLOR_CHANNELS = 1    # Number of color channels

VALID_SIZE = 1000     # Size of the Validation data

EPOCHS = 20000        # Number of epochs to run

BATCH = 32       # SGD Batch size

FILTER_SIZE = 5       # Filter size for kernel

DEPTH = 32            # Number of filters/templates

HIDDEN = 1024 #1024 # Number of hidden neurons in the fully connected layer

LR = 0.001            # Learning rate Alpha for SGD

PATCH = 5 # Convolutional Kernel size

STEPS = 5000# Number of steps to run
labels = mnist.pop('label')
mnist.head()
mnist = StandardScaler().fit_transform(np.float32(mnist.values))
mnist.shape
mnist = mnist.reshape(-1,WIDTH,WIDTH,COLOR_CHANNELS)
mnist.shape
train_data, valid_data = mnist[:-VALID_SIZE], mnist[-VALID_SIZE:]

train_labels, valid_labels = labels[:-VALID_SIZE], labels[-VALID_SIZE:]
#lets use keras -

from tensorflow import keras
#building the model

model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28, 28,1)),

    keras.layers.Dense(128, activation=tf.nn.relu),

    keras.layers.Dense(10, activation=tf.nn.softmax)

])



model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(valid_data, valid_labels)



print('Test accuracy:', test_acc)
predictions = model.predict(valid_data)
np.argmax(predictions[0])
#building the model

model = keras.Sequential([

    keras.layers.Conv2D(filters=32, kernel_size=5,input_shape=(28,28,1)),

    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Conv2D(filters=64, kernel_size=5),

    keras.layers.MaxPooling2D(pool_size=(2, 2)),

    keras.layers.Flatten(input_shape=(7, 7,64)),

    keras.layers.Dense(128, activation=tf.nn.relu),

    keras.layers.Dense(10, activation=tf.nn.softmax)

])



model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])


model.fit(train_data, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(valid_data, valid_labels)



print('Test accuracy:', test_acc)
#an increase of approx 3% accuracy using convolutional neural network