# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import keras

keras.__version__

#train = pd.read_csv('../input/mnist_train.csv', header=0)

#test = pd.read_csv('../input/mnist_test.csv', header=0)

test = np.genfromtxt('../input/mnist_test.csv', delimiter=',',dtype=None)

train = np.genfromtxt('../input/mnist_train.csv', delimiter=',',dtype=None)

test_labels = test[:,0]

train_labels = train[:,0]

test_images = test[:,1:].reshape(10000,28,28)

train_images = train[:,1:].reshape(60000,28,28)
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
network.compile(optimizer='rmsprop',

                loss='categorical_crossentropy',

                metrics=['accuracy'])
train_images = train_images.reshape((60000, 28 * 28))

train_images = train_images.astype('float32') / 255



test_images = test_images.reshape((10000, 28 * 28))

test_images = test_images.astype('float32') / 255
from keras.utils import to_categorical



train_labels = to_categorical(train_labels)

test_labels = to_categorical(test_labels)
network.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)