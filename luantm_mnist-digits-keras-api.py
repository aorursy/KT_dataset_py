import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input, Reshape, MaxPooling2D, Conv2D, Dense,Flatten
Data = np.load("../input/mnist.npz")
x_train = Data['x_train'] / 255
x_test = Data['x_test'] / 255

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

y_train_cls = Data['y_train']
y_train = np.zeros((y_train_cls.shape[0] , 10))
y_train[np.arange(y_train_cls.shape[0]), y_train_cls] = 1

y_test_cls = Data['y_test']
y_test = np.zeros((y_test_cls.shape[0] , 10))
y_test[np.arange(y_test_cls.shape[0]), y_test_cls] = 1
print(x_train.shape)
print(x_test.shape)
print(y_train)
print(y_train_cls.shape)
# The number of pixels in each dimension of an image.
img_size = 28

# The images are stored in one-dimensional arrays of this length.
img_size_flat = 28 * 28

# Tuple with height and width of images used to reshape arrays.
img_shape = (28, 28)

# Number of classes, one class for each of 10 digits.
num_classes = 10

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1
model = Sequential()
model.add(InputLayer(input_shape=( img_size, img_size, 1)))
model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
                activation='relu', name='layer_conv1'))
model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
                activation='relu', name='layer_conv2'))
model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
from tensorflow.python.keras.optimizers import Adam

optimizer = Adam(lr=1e-3)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=x_train,
          y=y_train,
          epochs=2, batch_size=128)
result = model.evaluate(x=x_test,
                        y=y_test)
print(result)
