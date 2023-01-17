from keras.datasets import mnist

import numpy as np

import pandas as pd

import math

import matplotlib,sys



(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1,784)

x_test = x_test.reshape(-1,784)

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255
m = 784

Weights = np.random.randn(m)

LR = 0.03

epochs = 10
def prediction(inputs,weights):

    sum = np.dot(inputs,weights)

    if sum >= 0: return 1

    else: return 0
for epoch in range(epochs):

    for sample in range(60000):

        Y_Pred = prediction(x_train[sample],Weights)

        error = y_train[sample] - Y_Pred

        Weights = Weights + (LR * error * x_train[sample])
Weights
# How can I Predict using Weight matrix now?