from keras.models import Input, Model

from keras.layers import Dense

from keras.datasets import fashion_mnist

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import json

import warnings

warnings.filterwarnings("ignore")
import os

print(os.listdir("../input"))
x_train = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")
x_test = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")
x_train = np.array(x_train)

x_test = np.array(x_test)
x_train = x_train.astype("float32")/255.0

x_test = x_test.astype("float32")/255.0

x_train = x_train[:,:-1]

x_test = x_test[:,:-1]

print(x_train.shape)

print(x_test.shape)
print(x_train.shape,x_test.shape)
plt.imshow(x_train[4000].reshape(28,28))

plt.show()

plt.imshow(x_train[1500].reshape(28,28))

plt.show()

plt.imshow(x_train[150].reshape(28,28))

plt.show()
input_img = Input(shape=(784,))

encoded = Dense(32, activation="relu")(input_img)

encoded = Dense(16, activation="relu")(encoded)
decoded = Dense(32, activation="relu")(encoded)

output_img = Dense(784, activation="sigmoid")(decoded)
autoencoder = Model(input_img, output_img)
autoencoder.compile(optimizer="rmsprop", loss="binary_crossentropy")
history = autoencoder.fit(x_train, x_train, 

                          epochs=200, 

                          batch_size=256, 

                          shuffle=True,

                          validation_data=(x_train, x_train))
plt.plot(history.history["loss"], label="Train Loss")

plt.plot(history.history["val_loss"], label="Validation Loss")

plt.legend()

plt.show()
x_test_pred = autoencoder.predict(x_test)
plt.imshow(x_test[100].reshape(28,28))

plt.title("Gerçek Resim")

plt.show()

plt.imshow(x_test_pred[100].reshape(28,28))

plt.title("Auto Encoder \nSonucu Çıkan Resim")

plt.show()
n = 10  # how many digits we will display

plt.figure(figsize=(20, 4))

for i in range(n):

    # display original

    ax = plt.subplot(2, n, i + 1)

    plt.imshow(x_test[i].reshape(28, 28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    # display reconstruction

    ax = plt.subplot(2, n, i + 1 + n)

    plt.imshow(x_test_pred[i].reshape(28, 28))

    plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()