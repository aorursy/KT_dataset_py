%matplotlib notebook

import matplotlib.pyplot as plt

import numpy as np

import time

# https://gist.github.com/greydanus/f6eee59eaf1d90fcb3b534a25362cea4

# https://stackoverflow.com/a/14434334

# this function is used to update the plots for each epoch and error

def plt_dynamic(fig,x, vy, ty, ax, colors=['b']):

    

    ax.plot(x, vy, 'b', label="Validation Loss")

    ax.plot(x, ty, 'r', label="Train Loss")

    plt.legend()

    plt.grid()

    fig.canvas.draw()
def printLossPlot(modelName):

    score = modelName.evaluate(x_test, y_test, verbose=0) 

    print('Test score:', score[0]) 

    print('Test accuracy:', score[1])

#     fig = plt.figure()

    fig,ax = plt.subplots(1,1)

    ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')



    # list of epoch numbers

    x = list(range(1,epochs+1))

    vy = history.history['val_loss']

    ty = history.history['loss']

    plt_dynamic(fig,x, vy, ty, ax)
# Credits: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py





from __future__ import print_function

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

from keras.layers.normalization import BatchNormalization



batch_size = 256

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



model1 = Sequential()

model1.add(Conv2D(32, kernel_size=(7, 7),activation='relu',input_shape=input_shape))

model1.add(Conv2D(64, (7, 7), activation='relu'))



model1.add(Conv2D(64, (7, 7), activation='relu'))



model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(BatchNormalization())

model1.add(Dropout(0.25))

model1.add(Flatten())



model1.add(Dense(128, activation='relu'))

model1.add(BatchNormalization())

model1.add(Dropout(0.5))



model1.add(Dense(num_classes, activation='softmax'))



model1.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])



history=model1.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(x_test, y_test))

score1 = model1.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score1[0])

print('Test accuracy:', score1[1])
printLossPlot(model1)
# Credits: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py





from __future__ import print_function

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

from keras.layers.normalization import BatchNormalization



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



model2 = Sequential()

model2.add(Conv2D(32, kernel_size=(7, 7),activation='relu',input_shape=input_shape))

model2.add(Conv2D(64, (5, 5), activation='relu'))



model2.add(MaxPooling2D(pool_size=(2, 2)))



model2.add(Conv2D(64, (3, 3), activation='relu'))

model2.add(MaxPooling2D(pool_size=(2, 2)))



model2.add(BatchNormalization())

model2.add(Dropout(0.25))

model2.add(Flatten())

model2.add(Dense(128, activation='relu'))

model2.add(BatchNormalization())

model2.add(Dropout(0.5))

model2.add(Dense(num_classes, activation='softmax'))



model2.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])



history=model2.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(x_test, y_test))

score2 = model2.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score2[0])

print('Test accuracy:', score2[1])
printLossPlot(model2)
# Credits: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py





from __future__ import print_function

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

from keras.layers.normalization import BatchNormalization



batch_size = 64

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



model3 = Sequential()

model3.add(Conv2D(32, kernel_size=(5, 5),

                 activation='relu',

                 input_shape=input_shape))





model3.add(Conv2D(64, (3, 3), activation='relu'))

model3.add(MaxPooling2D(pool_size=(2, 2)))

model3.add(BatchNormalization())

model3.add(Dropout(0.25))



model3.add(Conv2D(32, (3, 3), activation='relu'))

model3.add(MaxPooling2D(pool_size=(2, 2)))

model3.add(BatchNormalization())

model3.add(Dropout(0.25))



model3.add(Flatten())

model3.add(Dense(128, activation='relu'))

model3.add(BatchNormalization())

model3.add(Dropout(0.5))



model3.add(Dense(64, activation='relu'))

model3.add(BatchNormalization())

model3.add(Dropout(0.5))



model3.add(Dense(32, activation='relu'))

model3.add(BatchNormalization())

model3.add(Dropout(0.2))





model3.add(Dense(num_classes, activation='softmax'))



model3.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])



history=model3.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(x_test, y_test))

score3 = model3.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score3[0])

print('Test accuracy:', score3[1])
printLossPlot(model3)
from prettytable import PrettyTable

    

x = PrettyTable()



x.field_names = ["Model Name", "Test Score", "Test Accuracy"]



x.add_row(["Model1", score1[0], score1[1]])

x.add_row(["Model2", score2[0], score2[1]])

x.add_row(["Model3", score3[0],score3[1]])





print(x)