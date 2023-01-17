import numpy as np
import pandas as pd
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation = "relu"))
model.summary()
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))
model.summary()
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)
train_images = \
train_images.reshape((train_images.shape[0], train_images.shape[1], train_images.shape[2], 1))
test_images = \
test_images.reshape((test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))
train_images = train_images.astype("float32")
train_images = train_images / 255.0
test_images = test_images.astype("float32")
test_images = test_images / 255.0
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)
model.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(test_loss, test_accuracy)
model_without_maxpool = models.Sequential()
model_without_maxpool.add(layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)))
model_without_maxpool.add(layers.Conv2D(64, (3,3), activation="relu"))
model_without_maxpool.add(layers.Conv2D(64, (3,3), activation="relu"))

model_without_maxpool.summary()
model_without_maxpool.add(layers.Flatten())
model_without_maxpool.add(layers.Dense(64, activation="relu"))
model_without_maxpool.add(layers.Dense(10, activation="softmax"))
model_without_maxpool.summary()
model_without_maxpool.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
model_without_maxpool.fit(train_images, train_labels, epochs=5, batch_size=64)
test_loss, test_accuracy = model_without_maxpool.evaluate(test_images, test_labels)
print(test_loss, test_accuracy)