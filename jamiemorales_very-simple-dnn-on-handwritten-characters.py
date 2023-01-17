# Set-up libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
# Check source
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Load data
train_images = pd.read_csv('../input/emnist/emnist-balanced-train.csv')
test_images = pd.read_csv('../input/emnist/emnist-balanced-train.csv')

train_labels = np.array(train_images.iloc[:, 0].values)
train_images = np.array(train_images.iloc[:, 1:].values)

test_labels = np.array(test_images.iloc[:, 0].values)
test_images = np.array(test_images.iloc[:, 1:].values)
# Explore data
print('Shape of training data:', train_images.shape)
print('Shape of training labels:', train_labels.shape)

print('Shape of test data:', test_images.shape)
print('Shape of test labels:', test_labels.shape)
# Explore a few items
plt.figure(figsize=(12,12))
for i in range(1,21):
    plt.subplot(4,5,i)
    plt.imshow(train_images[i].reshape(28,28))
# Normalise and reshape data
train_images_number = train_images.shape[0]
train_images_height = 28
train_images_width = 28
train_images_size = train_images_height*train_images_width

train_images = train_images / 255.0
train_images = train_images.reshape(train_images_number, train_images_height, train_images_width, 1)

test_images_number = test_images.shape[0]
test_images_height = 28
test_images_width = 28
test_images_size = test_images_height*test_images_width

test_images = test_images / 255.0
test_images = train_images.reshape(test_images_number, test_images_height, test_images_width, 1)
# Transform labels
number_of_classes = 47

train_labels = tf.keras.utils.to_categorical(train_labels, number_of_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, number_of_classes)
# Explore data some more
print('Shape of training: ', train_images.shape)
print('Shape of training labels: ', train_images.shape)

print('Number of images: ', train_images_number)
print('Height of image: ', train_images_height)
print('Width of image: ', train_images_width)
print('Size of image: ', train_images_size)

print('\nShape of test: ', test_images.shape)
print('Shape of training labels: ', train_images.shape)

print('Number of images: ', test_images_number)
print('Height of image: ', test_images_height)
print('Width of image: ', test_images_width)
print('Size of image: ', test_images_size)
# Build and train neural network
model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28,28,1)),
                            tf.keras.layers.Dense(128, activation='relu'),
                            tf.keras.layers.Dense(64, activation='relu'),
                            tf.keras.layers.Dense(number_of_classes, activation='softmax')
                            ])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']
             )

model.fit(train_images, train_labels, epochs=15)
# Apply the neural network
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print('Test loss: {}, Test accuracy: {}'.format(test_loss, test_accuracy*100))
