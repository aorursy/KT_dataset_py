# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import struct
from array import array

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return np.array(images), np.array(labels)
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)        
training_images_filepath = '/kaggle/input/mnist-dataset/train-images.idx3-ubyte'
training_labels_filepath = '/kaggle/input/mnist-dataset/train-labels.idx1-ubyte'
test_images_filepath = '/kaggle/input/mnist-dataset/t10k-images.idx3-ubyte'
test_labels_filepath = '/kaggle/input/mnist-dataset/t10k-labels.idx1-ubyte'
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
x_train.shape
x_train = x_train/255.
x_test = x_test/255.
import matplotlib.pyplot as pt
for i in range(10):
    r = np.random.randint(60000)
    pt.imshow(x_train[r], cmap = pt.cm.gray)
    pt.title("Label : "+str(y_train[r]))
    
    
    
x_test.shape
from keras.utils import to_categorical
x_train = np.array([x.flatten() for x in x_train])
y_train = np.array([to_categorical(y, 10) for y in y_train])

x_test = np.array([x.flatten() for x in x_test])
y_test = np.array([to_categorical(y, 10) for y in y_test])
x_test.shape
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dropout
model = Sequential()
model.add(Convolution2D(128, kernel_size = (3, 3), input_shape = (28, 28, 1), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.1))

model.add(Convolution2D(64, kernel_size = (3, 3)))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.1))


model.add(Flatten())
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(25, activation = 'relu'))
model.add(Dropout(0.1))

model.add(Dense(10, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, 
         validation_data=(x_test, y_test), 
         epochs = 7)
score = model.history.history
pt.title("Epoch vs Loss")
pt.xlabel("Epoch")
pt.ylabel("Loss")
pt.plot(model.history.epoch, score['loss'], label = 'Training Loss')
pt.plot(model.history.epoch, score['val_loss'], label = 'Validation Loss')
pt.legend()
pt.show()

score = model.history.history
pt.title("Epoch vs Accuracy")
pt.xlabel("Epoch")
pt.ylabel("Accuracy")
pt.plot(model.history.epoch, score['accuracy'], label = 'Training Accuracy')
pt.plot(model.history.epoch, score['val_accuracy'], label = 'Validation Accuracy')
pt.legend()
pt.show()

model.evaluate(x_test, y_test)