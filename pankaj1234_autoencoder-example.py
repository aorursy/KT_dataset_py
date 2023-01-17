import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cv2

from convolutionalautoencoder import CNAutoEnc #import AutoEncoder utility script



%matplotlib inline
mtrain = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")

mtest = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")

mnist_train = np.array(mtrain)

mnist_test = np.array(mtest)

mnist_train = mnist_train[:,1:]

mnist_test = mnist_test[:,1:]

mnist_train=mnist_train.reshape(60000,28,28,1)

mnist_train = mnist_train.astype("float32")/255.0

mnist_test=mnist_test.reshape(10000,28,28,1)

mnist_test = mnist_test.astype("float32")/255.0

plt.imshow(mnist_train[0].reshape(28,28), cmap='gray')
epoch = 20

batch_size = 32

(encoder, decoder, autoencoder) = CNAutoEnc.create((28, 28, 1))

autoencoder.compile(loss="mse", optimizer="Adam")
history = autoencoder.fit(mnist_train, mnist_train,validation_data=(mnist_test,mnist_test),epochs=epoch,batch_size=batch_size)
recon = autoencoder.predict(mnist_test)

# Sample1

reconstructed = (recon[0]*255.0).astype("uint8")

test_sample = (mnist_test[0]*255.0).astype("uint8")

print("Reconstructed Image")

plt.imshow(reconstructed.reshape(28,28), cmap="gray")

print("Original Image")

plt.imshow(test_sample.reshape(28,28), cmap="gray")