# Credits: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py





from __future__ import print_function

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K



batch_size = 128

num_classes = 10

epochs = 12



# input image dimensions

img_rows, img_cols = 28, 28



# the data, split between train and test sets

(x_train, y_train), (x_test, y_test) = mnist.load_data()



if K.image_data_format() == 'channels_first':

    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)

    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)

    input_shape = (1, img_rows, img_cols)

else:

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    input_shape = (img_rows, img_cols, 1)



x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255

print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')



# convert class vectors to binary class matrices

y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)



%matplotlib notebook

import matplotlib.pyplot as plt

import numpy as np

import time

# https://gist.github.com/greydanus/f6eee59eaf1d90fcb3b534a25362cea4

# https://stackoverflow.com/a/14434334

# this function is used to update the plots for each epoch and error

def plt_dynamic(x, vy, ty, ax, colors=['b']):

    ax.plot(x, vy, 'b', label="Validation Loss")

    ax.plot(x, ty, 'r', label="Train Loss")

    plt.legend()

    plt.grid()

    fig.canvas.draw()

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])



history_2 = model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])

fig,ax = plt.subplots(1,1)

ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')



# list of epoch numbers

x = list(range(1,epochs+1))



# print(history.history.keys())

# dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])

# history = model_drop.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))



# we will get val_loss and val_acc only when you pass the paramter validation_data

# val_loss : validation loss

# val_acc : validation accuracy



# loss : training loss

# acc : train accuracy

# for each key in histrory.histrory we will have a list of length equal to number of epochs



vy = history_2.history['val_loss']

ty = history_2.history['loss']

plt_dynamic(x, vy, ty, ax)
model = Sequential()

model.add(Conv2D(32,kernel_size=(5,5),

                activation='relu',

                input_shape=input_shape))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(2,2), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.50))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])



history_3 = model.fit(x_train,y_train,

                   batch_size=batch_size,

                   epochs=epochs,

                   verbose=1,

                   validation_data=(x_test,y_test))

score= model.evaluate(x_test,y_test, verbose=0)

print('Test Loss:', score[0])

print('Test accuracy:', score[1])
fig,ax = plt.subplots(1,1)

ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')



# list of epoch numbers

x = list(range(1,epochs+1))



vy = history_3.history['val_loss']

ty = history_3.history['loss']

plt_dynamic(x, vy, ty, ax)
model = Sequential()

model.add(Conv2D(32,kernel_size=(5,5),

                activation='relu',

                input_shape=input_shape))

model.add(Conv2D(64, kernel_size=(5,5), activation='relu'))

model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(384, kernel_size=(2,2), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.50))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])



history_5 = model.fit(x_train,y_train,

                   batch_size=batch_size,

                   epochs=epochs,

                   verbose=1,

                   validation_data=(x_test,y_test))

score= model.evaluate(x_test,y_test, verbose=0)

print('Test Loss:', score[0])

print('Test accuracy:', score[1])

# padding='valid',
fig,ax = plt.subplots(1,1)

ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')



# list of epoch numbers

x = list(range(1,epochs+1))



vy = history_5.history['val_loss']

ty = history_5.history['loss']

plt_dynamic(x, vy, ty, ax)
model = Sequential()

model.add(Conv2D(32,kernel_size=(5,5),

                activation='relu',

                input_shape=input_shape))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))

model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))

model.add(Conv2D(384, kernel_size=(2,2), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.10))

# model.add(Conv2D(384, kernel_size=(2,2), activation='relu'))

# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Dropout(0.10))

# model.add(Conv2D(512, kernel_size=(2,2), activation='relu'))

# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.50))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])



history_7 = model.fit(x_train,y_train,

                   batch_size=batch_size,

                   epochs=epochs,

                   verbose=1,

                   validation_data=(x_test,y_test))

score= model.evaluate(x_test,y_test, verbose=0)

print('Test Loss:', score[0])
fig,ax = plt.subplots(1,1)

ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')



# list of epoch numbers

x = list(range(1,epochs+1))



vy = history_7.history['val_loss']

ty = history_7.history['loss']

plt_dynamic(x, vy, ty, ax)