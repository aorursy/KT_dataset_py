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



history =  model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
def plt_dynamic(x, vy, ty, ax, colors=['b']):

    ax.plot(x, vy, 'b', label="Validation Loss")

    ax.plot(x, ty, 'r', label="Train Loss")

    plt.legend()

    plt.grid()

    fig.canvas.draw()
import matplotlib.pyplot as plt



fig,ax = plt.subplots(1,1)

ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')



# list of epoch numbers

x = list(range(1,epochs+1))



vy = history.history['val_loss']

ty = history.history['loss']

plt_dynamic(x, vy, ty, ax)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape, padding='same'))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Conv2D(128, kernel_size=(3, 3),activation='relu', padding='same'))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])



history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,

          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
fig,ax = plt.subplots(1,1)

ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')



# list of epoch numbers

x = list(range(1,epochs+1))



vy = history.history['val_loss']

ty = history.history['loss']

plt_dynamic(x, vy, ty, ax)
model = Sequential()

model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape, padding='same'))

model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Conv2D(96, kernel_size=(5, 5),activation='relu', padding='same'))

model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))

model.add(Conv2D(224, kernel_size=(5, 5),activation='relu', padding='same'))

model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])



history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,

          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
fig,ax = plt.subplots(1,1)

ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')



# list of epoch numbers

x = list(range(1,epochs+1))



vy = history.history['val_loss']

ty = history.history['loss']

plt_dynamic(x, vy, ty, ax)
model = Sequential()

model.add(Conv2D(32, kernel_size=(7, 7), activation='relu', input_shape=input_shape, padding='same'))

model.add(Conv2D(64, (7, 7), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Conv2D(96, kernel_size=(7, 7),activation='relu', padding='same'))

model.add(Conv2D(128, (7, 7), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Conv2D(224, kernel_size=(7, 7),activation='relu', padding='same'))

model.add(Conv2D(512, kernel_size=(7, 7),activation='relu', padding='same'))

model.add(Conv2D(512, kernel_size=(7, 7),activation='relu', padding='same'))

model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])



history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,

          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
fig,ax = plt.subplots(1,1)

ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')



# list of epoch numbers

x = list(range(1,epochs+1))



vy = history.history['val_loss']

ty = history.history['loss']

plt_dynamic(x, vy, ty, ax)
from prettytable import PrettyTable



table = PrettyTable()

table.field_names = ['# Conv Layers', 'Filter Size', 'Test Loss', 'Test Accuracy']

table.add_row([3, '(3, 3)', 0.0259, 0.992])

table.add_row([5, '(5, 5)', 0.0187, 0.9955])

table.add_row([7, '(7, 7)', 0.0237, 0.994])

print(table)