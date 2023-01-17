# Many libraries like Keras, Scikit-learn, NLTK has nicely packaged some

# popular datasets into some Pythonic readers. We often overlook how

# the data came to be and undervalue the simply of doing `.load_data()` function.

from keras.datasets import mnist



"""

# Whoops, hit a Kaggle Kernel wall when we do this:



    >>> train_images, train_labels), (test_images, test_labels) = mnist.load_data()



# Let's go report this on the product feedback in the discussion! 

"""



# Meanwhile someone has uploaded the data on 

# https://www.kaggle.com/apallekonda/keras-mnist

# So lets use that first but we have to do some weird stuns 

# by writing our own mnist_load_data() function.

import numpy as np

def mnist_load_data(path='mnist.npz'):

    with np.load(path) as f:

        x_train, y_train = f['x_train'], f['y_train']

        x_test, y_test = f['x_test'], f['y_test']

    return (x_train, y_train), (x_test, y_test)

        



(train_images, train_labels), (test_images, test_labels) = mnist_load_data(path='../input/mnist.npz')



print(train_images.shape)
from keras.utils import to_categorical



# Currently our data has a shape of (60000, 28, 28), 

# which means, there are 60,000 data points (i.e. images of handwritting)

# and each data point is a 28x28 pixel image represented as a 28x28 matrix 

# where each value in the matrix is the "blackness" of the pixel.



# So for simplicity, we can transform the 28x28 matrix into a simple 28*28 vector.

train_images = train_images.reshape((60000, 28 * 28))

train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))

test_images = test_images.astype('float32') / 255



# And for keras sake, they have a neat interface to indicate

# where the prediction is categorical or a discrete number.

train_labels = to_categorical(train_labels)

test_labels = to_categorical(test_labels)
from keras import models

from keras import layers

network = models.Sequential()

network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))

network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop',

loss='categorical_crossentropy',

metrics=['accuracy'])
network.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)

print('test_acc:', test_acc)