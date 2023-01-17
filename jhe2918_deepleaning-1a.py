import keras
#import Keras libray
keras.__version__
print('hello, world')
print("Welcome to use Jupyter Notebook")
print("hello, Liam, James, Moham")
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = train_images[0]

import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
train_images.shape
len(train_labels)
train_labels
test_images.shape
len(test_labels)
test_labels
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
import numpy as np
z = [1.0, 2.0, 3.0, 7.0, 1.0, 1.0, 2.0, 3.0, 4.0, 1.0]
np.exp(z) / np.sum(np.exp(z))  
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
#digit = train_images[4]

#import matplotlib.pyplot as plt
#plt.imshow(digit, cmap=plt.cm.binary)
#plt.show()
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
train_labels
test_labels
network.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)