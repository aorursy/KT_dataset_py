%matplotlib inline

import matplotlib.pyplot as plt # graphics plotting

import random # random number generation

import numpy as np # linear algebra

import keras as ks # deep learning

from time import time
# Load the training and test datasets

(train_images, train_labels), (test_images, test_labels) = ks.datasets.mnist.load_data()



# Print the shape of the training and test image tensors

print("train_images.shape = {}".format(train_images.shape))

print("train_labels.shape = {}".format(train_labels.shape))

print("test_images.shape = {}".format(test_images.shape))

print("test_labels.shape = {}".format(test_labels.shape))



# Select a training image and label at random and display them

index = random.randint(0,train_images.shape[0])

print("\nlabel = {}".format(train_labels[index]))

plt.imshow(train_images[index], cmap=plt.cm.binary)

plt.show()
# Flatten the images into vectors

train_images = train_images.reshape(60000, 28*28)

test_images = test_images.reshape(10000, 28*28)



# Convert the one byte pixel values into 4-byte floats

train_images = train_images.astype('float32')

test_images = test_images.astype('float32')



# Normalize the pixel values (0..255) to range from 0.0 to 1.0 

train_images /= 255

test_images /= 255



# Convert the labels from 0..9 to one-hot float32 vectors

train_labels = ks.utils.to_categorical(train_labels, 10)

test_labels = ks.utils.to_categorical(test_labels, 10)



print("train_images.shape = {}".format(train_images.shape))

print("train_labels.shape = {}".format(train_labels.shape))

print("test_images.shape = {}".format(test_images.shape))

print("test_labels.shape = {}".format(test_labels.shape))
# Define the layers

network = ks.models.Sequential()

network.add(ks.layers.Dense(512, activation='relu', input_shape=(784,)))

network.add(ks.layers.Dense(10, activation='softmax'))

network.summary()



# Specify the loss function, optimization approach, and monitored metrics

network.compile(loss='categorical_crossentropy', optimizer='RMSProp', metrics=['accuracy'])
start = time()

network.fit(train_images, train_labels, epochs=10, batch_size=128)

end = time()



print("\nTime to fit model: {} seconds".format(round(end-start)))
score = network.evaluate(test_images, test_labels, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])