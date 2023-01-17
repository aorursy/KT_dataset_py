import numpy as np

import pandas as pd



import matplotlib.pyplot as plt



from PIL import Image



from keras.layers import Dense, Conv2D, Flatten, MaxPool2D

from keras.models import Sequential

from keras.utils import to_categorical



import os



import random

random.seed(0)
PATH = '../input/notmnist/notMNIST_small/notMNIST_small'



classes = os.listdir(PATH)

num_classes = len(classes)



print("There are {} classes: {}".format(num_classes, classes))
X = []

y = []



for directory in os.listdir(PATH):

    for image in os.listdir(PATH + '/' + directory):

        try:

            path = PATH + '/' + directory + '/' + image

            img = Image.open(path)

            img.load()

            img_X = np.asarray(img, dtype=np.int16)

            X.append(img_X)

            y.append(directory)

        except:

            None

            

X = np.asarray(X)

y = np.asarray(y)
num_images = len(X)

size = len(X[0])



X.shape, y.shape
plt.imshow(X[0], cmap='gray')

plt.title("Letter {}".format(y[0]));
for let in sorted(classes):

    letter = X[y == let] 



    plt.figure(figsize=(15,20))

    for i in range(5):

        plt.subplot(10, 5, i+1)

        plt.imshow(letter[i], cmap='gray')

        plt.title("Letter {}".format(let))

    

plt.tight_layout()

plt.show();
y = list(map(lambda x: ord(x) - ord('A'), y))

y = np.asarray(y)
indices = np.arange(X.shape[0])

np.random.shuffle(indices)



X = X[indices]

y = y[indices]
num_train_img = np.int(0.7 * X.shape[0])

num_train_img
X_train = X[:num_train_img]

y_train = y[:num_train_img]



X_test = X[num_train_img:]

y_test = y[num_train_img:]
if y_train.ndim == 1: y_train = to_categorical(y_train, 10)

if y_test.ndim == 1: y_test = to_categorical(y_test, 10)
if np.max(X_train) == 255: X_train = X_train / 255 

if np.max(X_test) == 255: X_test = X_test / 255
if X_train.ndim == 3:

    num_pixels = size * size

    X_train_1d = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')

    X_test_1d = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
if X_train.ndim == 3:

    X_train = X_train.reshape(-1, size, size, 1).astype('float32')

    X_test = X_test.reshape(-1, size, size, 1).astype('float32')
X_train.shape, X_test.shape
input_shape = X_train.shape[1:]

input_shape
X_train_1d.shape, X_test_1d.shape
input_shape_1d = X_train_1d.shape[1]

input_shape_1d
y_train.shape, y_test.shape
model_mlp = Sequential([

    Dense(512, input_dim=input_shape_1d, activation='relu'),

    Dense(256, activation='relu'),

    Dense(128, activation='relu'),

    Dense(num_classes, activation='softmax')

    

])



model_mlp.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')

model_mlp.summary()
model_mlp.fit(X_train_1d, y_train, epochs=10)
score = model_mlp.evaluate(X_test_1d, y_test, verbose=1)
model_cnn = Sequential([

    Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=input_shape),

    MaxPool2D(),

    

    Conv2D(filters=64, kernel_size=(3,3), activation='relu'),

    MaxPool2D(),

    

    Flatten(),

    

    Dense(512, activation='relu'),

    Dense(num_classes, activation='softmax'),

])



model_cnn.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')

model_cnn.summary()
model_cnn.fit(X_train, y_train, epochs=10)
score = model_cnn.evaluate(X_test, y_test, verbose=1)