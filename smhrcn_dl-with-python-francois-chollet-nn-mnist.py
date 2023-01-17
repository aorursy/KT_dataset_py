#Deep Learning with Python: Francois Chollet - NN - MNIST

#2.1 Sinir Ağlarına İlk Bakış



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import keras



from keras.datasets import mnist

from keras import models

from keras import layers

from keras.utils import to_categorical
#2.1 Keras'ta MNIST veri setini yükleme

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()



print('train_images shape:', train_images.shape)

print('train_labels shape:',train_labels.shape)

print('test_images shape:', test_images.shape)

print('test_labels shape:',test_labels.shape)



#2.6 Veri setinden bir örneğin gösterimi

digit = train_images[4]

plt.imshow(digit, cmap=plt.cm.binary)

plt.show()
#2.2 Ağ Mimarisi

network = models.Sequential()

network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))

network.add(layers.Dense(10, activation='softmax'))



#2.3 Derleme

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#2.4 Girdilerin Hazırlanması

train_images = train_images.reshape((60000, 28 * 28))

train_images = train_images.astype('float32') / 255



test_images = test_images.reshape((10000, 28 * 28))

test_images = test_images.astype('float32') / 255
#2.5 Etiketlerin Hazırlanması

train_labels = to_categorical(train_labels)

test_labels = to_categorical(test_labels)



network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)

print('test_acc', test_acc)
