# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import seaborn as sns

import tensorflow as tf

import torch

from matplotlib import pyplot as plt

from sklearn import datasets, neural_network

from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Activation



model = Sequential()

model.add(Dense(64, activation='relu', input_dim=100))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',

              loss='binary_crossentropy',

              metrics=['accuracy'])



# Generate dummy data

import numpy as np

data = np.random.random((1000, 100))

labels = np.random.randint(2, size=(1000, 1))



# Train the model, iterating on the data in batches of 32 samples

model.fit(data, labels, epochs=20, batch_size=1)
model1 = Sequential()

model1.add(Dense(64, activation='relu', input_dim=100))

model1.add(Dense(1, activation='sigmoid'))

model1.compile(optimizer='sgd',

              loss='binary_crossentropy',

              metrics=['accuracy'])

model1.fit(data, labels, epochs=20, batch_size=1)
model2 = Sequential()

model2.add(Dense(64, activation='relu', input_dim=100))

model2.add(Dense(1, activation='sigmoid'))

model2.compile(optimizer='adagrad',

              loss='binary_crossentropy',

              metrics=['accuracy'])

model2.fit(data, labels, epochs=20, batch_size=1)