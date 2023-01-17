#Importing the relevant modules

import struct

import random

import numpy as np 

import pandas as pd

import tensorflow as tf

from array import array

from os.path  import join

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
class MnistDataloader(object):

    def __init__(self, training_images_filepath,training_labels_filepath,

                 test_images_filepath, test_labels_filepath):

        self.training_images_filepath = training_images_filepath

        self.training_labels_filepath = training_labels_filepath

        self.test_images_filepath = test_images_filepath

        self.test_labels_filepath = test_labels_filepath

    

    def read_images_labels(self, images_filepath, labels_filepath):        

        labels = []

        with open(labels_filepath, 'rb') as file:

            magic, size = struct.unpack(">II", file.read(8))

            if magic != 2049:

                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))

            labels = array("B", file.read())        

        

        with open(images_filepath, 'rb') as file:

            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))

            if magic != 2051:

                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))

            image_data = array("B", file.read())        

        images = []

        for i in range(size):

            images.append([0] * rows * cols)

        for i in range(size):

            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])

            img = img.reshape(28, 28)

            images[i][:] = img            

        

        return images, labels

            

    def load_data(self):

        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)

        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)

        return (x_train, y_train),(x_test, y_test)        
#Storing the path of image files in 4 separate variables for further use.

X_train_path = "/kaggle/input/mnist-dataset/train-images.idx3-ubyte"

y_train_path = "/kaggle/input/mnist-dataset/train-labels.idx1-ubyte"

X_test_path = "/kaggle/input/mnist-dataset/t10k-images.idx3-ubyte"

y_test_path = "/kaggle/input/mnist-dataset/t10k-labels.idx1-ubyte"
mnist_dataloader = MnistDataloader(X_train_path, y_train_path, X_test_path, y_test_path)

(X_train, y_train), (X_test, y_test) = mnist_dataloader.load_data()
X_train = np.array([np.ravel(x) for x in X_train])

y_train = np.array([np.ravel(x) for x in y_train])

X_test = np.array([np.ravel(x) for x in X_test])

y_test = np.array([np.ravel(x) for x in y_test])



X_train = X_train.reshape((60000, 28, 28))

X_test = X_test.reshape((10000, 28, 28))
X_train_scaled , X_test_scaled = (X_train/255, X_test/255)
X_train_scaled = X_train_scaled[...,np.newaxis]

X_test_scaled = X_test_scaled[...,np.newaxis]
def getModel(shape):

    

    model = Sequential([

        Conv2D(8, (3,3), activation="relu", padding = "SAME", input_shape = shape),

        MaxPool2D((2,2)),

        Flatten(),

        Dense(64, activation="relu"),

        Dense(64, activation="relu"),

        Dense(10, activation="softmax")

    ])

    

    return model
#Getting an instance of the model 

model = getModel(X_train_scaled[0].shape)
opt = tf.keras.optimizers.Adam()

accuracy = tf.keras.metrics.SparseCategoricalAccuracy()



model.compile(optimizer = opt, loss = "sparse_categorical_crossentropy", metrics = [accuracy])
print(model.optimizer)

print(model.loss)

print(model.metrics)
result = model.fit(X_train_scaled, y_train, epochs = 5, verbose = 2)
dfMetrics = pd.DataFrame(result.history)

dfMetrics
dfMetrics.plot(y="loss", title = "Loss vs epochs");
dfMetrics.plot(y="sparse_categorical_accuracy", title = "Accuracy vs epochs");
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=2)
print("Test Accuracy : " + str(round(test_accuracy,4)))