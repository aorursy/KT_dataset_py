# Set-up libraries
import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
# Load data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# Explore data
train_images_number = train_images.shape[0]
train_images_height = train_images.shape[1]
train_images_width = train_images.shape[2]
train_images_size = train_images_height * train_images_width

print('Number of training images: ', train_images_number)
print('Height of training image: ', train_images_height)
print('Width of training image: ', train_images_width)
print('Size of training image: ',train_images_size)

test_images_number = test_images.shape[0]
test_images_height = test_images.shape[1]
test_images_width = test_images.shape[2]
test_images_size = test_images_height * test_images_width

print('\nNumber of training images: ', test_images_number)
print('Height of training image: ', test_images_height)
print('Width of training image: ', test_images_width)
print('Size of training image: ',test_images_size)
# Explore data visually
train_labels_categorical = ['T-shirt/top', 
'Trouser',
'Pullover',
'Dress',
'Coat',
'Sandal',
'Shirt',
'Sneaker',
'Bag',
'Ankle boot']

plt.subplots(figsize=(12,12))
for i in range(1,10):
    random_item = random.randint(1, train_images_number)
    plt.subplot(3, 3, i)
    plt.xlabel(train_labels_categorical[train_labels[random_item]])
    plt.imshow(train_images[random_item])
# Reshape images
train_images = train_images.reshape(train_images_number, train_images_height, train_images_width, 1)
test_images = test_images.reshape(test_images_number, test_images_height, test_images_width, 1)
# Normalise
train_images = train_images / 255.0
test_images = test_images / 255.0
# Build and train neural network
model = tf.keras.Sequential([
    keras.layers.Conv2D(64, (3,3), 
                       activation='relu',
                       input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(), 
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile  neural network
model.compile(optimizer=tf.keras.optimizers.Adam(),
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy']
             )

# Train the  neural network
model.fit(train_images, train_labels, epochs=5)
# Apply the neural network
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print('Test loss: {}, Test accuracy: {}'.format(test_loss, test_accuracy*100))
