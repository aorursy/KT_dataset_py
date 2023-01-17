# Set-up libraries
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
# Check source
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Load data
images = np.load('../input/sign-language-digits-dataset/Sign-language-digits-dataset/X.npy')
labels = np.load('../input/sign-language-digits-dataset/Sign-language-digits-dataset/Y.npy')
# Explore data
images_number = images.shape[0]
images_height = 64
images_width = 64
images_size = images_height * images_width

print('Number of images: ', images_number)
print('Height of image: ', images_height)
print('Width of image: ', images_width)
print('Size of image: ', images_size)

print('Shape of dataset: ', images.shape)
print('Shape of labels: ', labels.shape)
# Explore a few items
plt.figure(figsize=(12,12))

for i in range(1,13):
    random_item = random.randint(1, images_number)
    plt.subplot(3, 4, i)
    plt.imshow(images[random_item])
# Reshape data
images = images.reshape(images_number, images_height, images_width, 1)
# Normalise data
images = images / 255.0
# Split data into 80% train and 20% 
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=0)
# Explore processed data
train_images_number = train_images.shape[0]
train_images_height = train_images.shape[1]
train_images_width = train_images.shape[2]
train_images_size = train_images_height * train_images_width

print('Number of images: ', train_images_number)
print('Height of image: ', train_images_height)
print('Width of image: ', train_images_width)
print('Size of image: ', train_images_size)

print('Shape of training data: ', train_images.shape)
print('Shape of training labels: ', train_labels.shape)

test_images_number = test_images.shape[0]
test_images_height = test_images.shape[1]
test_images_width = test_images.shape[2]
test_images_size = test_images_height * test_images_width

print('\nNumber of images: ', test_images_number)
print('Height of image: ', test_images_height)
print('Width of image: ', test_images_width)
print('Size of image: ', test_images_size)

print('Shape of training data: ', test_images.shape)
print('Shape of training labels: ', test_labels.shape)
# Build and train neural network
model = tf.keras.Sequential([
        keras.layers.Conv2D(64, (5,5), padding = 'same', activation='relu', input_shape=(64, 64, 1)),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

# Compile neural network
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy']
            )

# Train neural network
model.fit(train_images, train_labels, epochs=5, use_multiprocessing=True)
# Apply the neural network
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print('Test loss: {}, Test accuracy: {}'.format(test_loss, test_accuracy*100))
