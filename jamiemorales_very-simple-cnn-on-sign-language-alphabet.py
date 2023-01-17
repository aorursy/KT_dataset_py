# Set-up libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelBinarizer
# Check source
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Load data
train_images = pd.read_csv('../input/sign-language-mnist/sign_mnist_train.csv')
test_images = pd.read_csv('../input/sign-language-mnist/sign_mnist_test.csv')

train_labels = np.array(train_images['label'].values)
train_images = np.array(train_images.drop('label', axis=1).values)

test_labels = np.array(test_images['label'].values)
test_images = np.array(test_images.drop('label', axis=1).values)
# Explore a few items
plt.figure(figsize=(12,12))
for i in range(1,21):
    plt.subplot(4,5,i)
    plt.imshow(train_images[i].reshape(28,28))
# Reshape and normalise data
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
test_images = test_images.reshape(test_images_number, test_images_height, test_images_width, 1)

# Transform labels
lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
test_labels = lb.fit_transform(test_labels)
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
model = tf.keras.Sequential([
    keras.layers.Conv2D(64, (8,8), padding='same', activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(24, activation='softmax')
])

# Compile neural network
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy']
             )

# Train the neural network
model.fit(train_images, train_labels, epochs=5)
# Apply the neural network
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print('Test loss: {}, Test accuracy: {}'.format(test_loss, test_accuracy*100))