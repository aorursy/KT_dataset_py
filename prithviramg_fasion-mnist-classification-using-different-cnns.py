from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import tensorflow as tf

tf.test.gpu_device_name()
import keras

from keras.models import Sequential

from keras.optimizers import adam

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Dense, Dropout, Flatten

from keras import backend

from keras.datasets import fashion_mnist

from keras.layers.normalization import BatchNormalization



import numpy as np

import matplotlib.pyplot as plt

import random as rand
((x_train, y_train), (x_test, y_test)) = fashion_mnist.load_data()
print('train images :',x_train.shape)

print('train images labels :',y_train.shape)

print('test images :',x_test.shape)

print('test images labels :',y_test.shape)
plt.figure(figsize=(20,10))

for i in range(1,66):

    plt.subplot(5,13,i)

    idx = rand.randint(1,60000)

    grid_data = x_train[idx]

    plt.imshow(grid_data, interpolation="none",cmap="gray")

    plt.title(str(y_train[idx]))

plt.suptitle('Displaying random images and their class number')

plt.show()

    
if backend.image_data_format() == 'channels_first':

    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)

    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

else:

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255

y_train = keras.utils.to_categorical(y_train,10)

y_test = keras.utils.to_categorical(y_test,10)
model = Sequential()

#input layer

model.add(Conv2D(128, kernel_size=(5, 5),activation='relu',input_shape=(28,28,1)))



#hidden layer1

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



#hidden layer2

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



#Batch Normalization layer

model.add(BatchNormalization())



#hidden layer3 flattened

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))



#output layer

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=adam(),

              metrics=['accuracy'])



details = model.fit(x_train, y_train,

          batch_size=1000,

          epochs=60,

          verbose=1,

          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
details
plt.plot(details.history['loss'])

plt.plot(details.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Loss in train dataset', 'Loss in Test dataset'], loc='upper right')

plt.show()
plt.plot(details.history['acc'])

plt.plot(details.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train Accuracy', 'Test Accuracy'], loc='lower right')

plt.show()