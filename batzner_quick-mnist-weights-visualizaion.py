from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
# Dataset preprocessing
train = pd.read_csv("../input/train.csv")
train, valid = train_test_split(train, test_size=0.2)
X_train = (train.iloc[:,1:].values).reshape(33600, 784).astype('float32') / 255
y_train = train.iloc[:,0].values.astype('int32')
y_train = to_categorical(y_train, 10)
X_valid = (valid.iloc[:,1:].values).reshape(8400, 784).astype('float32') / 255
y_valid = valid.iloc[:,0].values.astype('int32')
y_valid = to_categorical(y_valid, 10)
model = Sequential([
  Dense(100, input_shape=(784,)),
  Activation('relu'),
  Dense(50),
  Activation('relu'),
  Dense(10),
  Activation('softmax')
])

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=[X_valid, y_valid], epochs=10)
# Get the learned weights
dense_layers = [l for l in model.layers if l.name.startswith('dense')]
kernels, biases = zip(*[l.get_weights() for l in dense_layers])
print([k.shape for k in kernels])
print([b.shape for b in biases])
# Visualize the digits
%matplotlib inline
plt.figure(figsize=(12,5))
x, y = 5, 2
for digit in range(10):
    triggers = kernels[0].dot(kernels[1]).dot(kernels[2])[:, digit]
    triggers = triggers.reshape(28, 28) / np.absolute(triggers).max() * 255
    # Make the base image black
    pixels = np.full((28, 28, 3), 0, dtype=np.uint8)
    # Color positive values green
    green = np.clip(triggers, a_min=0, a_max=None)
    pixels[:, :, 1] += green.astype(np.uint8)
    # Color negative values red
    red = -np.clip(triggers, a_min=None, a_max=0)
    pixels[:, :, 0] += red.astype(np.uint8)

    plt.subplot(y, x, digit+1)
    plt.imshow(pixels)
plt.show()
# Visualize the first 20 neurons in tsecond hidden layer
%matplotlib inline
plt.figure(figsize=(12,10))
x, y = 5, 4
for neuron in range(20):
    triggers = kernels[0].dot(kernels[1])[:, neuron]
    triggers = triggers.reshape(28, 28) / np.absolute(triggers).max() * 255
    # Make the base image black
    pixels = np.full((28, 28, 3), 0, dtype=np.uint8)
    # Color positive values green
    green = np.clip(triggers, a_min=0, a_max=None)
    pixels[:, :, 1] += green.astype(np.uint8)
    # Color negative values red
    red = -np.clip(triggers, a_min=None, a_max=0)
    pixels[:, :, 0] += red.astype(np.uint8)

    plt.subplot(y, x, neuron+1)
    plt.imshow(pixels)
plt.show()