import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from numpy import genfromtxt

from PIL import Image

import csv

from time import time

from sklearn.model_selection import train_test_split





my_lable = genfromtxt('../input/finaldataset/dataset.csv', delimiter=';',usecols = 0)

#print(my_data)



my_data = genfromtxt('../input/finaldataset/dataset.csv', delimiter=';',usecols = range(1,401))

type(my_data)

#print(my_data1)

X_train, X_test, y_train, y_test = train_test_split(my_data,my_lable , test_size=0.2, random_state=42)

print(X_train)

print(X_test)

print(y_train)

print(y_test)

print("Validation dataset shape: {0}, \nTest dataset shape: {1}".format(X_train.shape, X_test.shape))
import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt
y_train = y_train.reshape((19200,1))

print(y_train.shape)
print(X_train[0].shape)

print(y_train[0].shape)

model = keras.Sequential()



#linear

#exponential

#sigmoid

#elu

#relu





model.add(keras.layers.Dense(100)) # скрытый слой из 100 нейронов

model.add(keras.layers.Activation('selu')) # функция активации



model.add(keras.layers.Dense(100)) # скрытый слой из 100 нейронов

model.add(keras.layers.Activation('elu')) # функция активации



model.add(keras.layers.Dense(100)) # скрытый слой из 100 нейронов

model.add(keras.layers.Activation('selu')) # функция активации 



model.add(keras.layers.Dense(54)) # 54 - число классов

model.add(keras.layers.Activation('softmax')) # функция активации

model.compile(optimizer=keras.optimizers.SGD(lr=0.01), 

                     loss='sparse_categorical_crossentropy',

                     metrics=['accuracy'])





tic = time()

history = model.fit( X_train,y_train, epochs=50, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test,y_test)

toc = time()

print('Test accuracy:', test_acc, '\nTest loss:', test_loss)

print(toc - tic)
test_loss, test_acc = model.evaluate(X_test,y_test)

print('Test accuracy:', test_acc, '\nTest loss:', test_loss)
# Plot training & validation accuracy values

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

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
model.summary()
from keras.utils import plot_model

plot_model(model, show_shapes = True, show_layer_names = True,expand_nested = True, to_file='model.png')