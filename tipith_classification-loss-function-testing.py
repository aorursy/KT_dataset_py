# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
import matplotlib.pyplot as plt

plt.imshow(x_train[2], cmap='RdBu')
x_train[0].shape
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.utils import to_categorical

model = Sequential()
model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
x_train2 = x_train[:,:,:,np.newaxis]
x_test2 = x_test[:,:,:,np.newaxis]
y_train[0:10]
history = model.fit(x_train2, to_categorical(y_train), 
                    batch_size=64, epochs=4, 
                    validation_data=(x_test2, to_categorical(y_test)))
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
test_ind = 1212

pred = model.predict(x_test2[test_ind:test_ind+1], verbose=1)
print(pred[0], np.sum(pred[0]), np.argmax(pred[0]))
plt.imshow(x_test[test_ind])
from keras.utils import plot_model
plot_model(model, to_file='model.png')
import cv2
plt.rcParams['figure.figsize'] = (10.0, 20.0)
img = cv2.imread('model.png')
plt.imshow(img)