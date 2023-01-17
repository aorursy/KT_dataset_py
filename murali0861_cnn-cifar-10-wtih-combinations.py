from __future__ import print_function

import keras

from keras.datasets import cifar10

from keras.models import Sequential

from keras.layers import Dropout, Dense, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.regularizers import l2

import numpy as np

import os

import matplotlib.pyplot as plt
batch_size = 32

epochs = 50

num_classes = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# =====> Take only part of data

# idx = np.random.randint(x_train.shape[0], size=20000)

# x_train = x_train[idx, :]

# y_train = y_train[idx]
print("Training data size X: ", x_train.shape, " Y train: ", y_train.shape)

print("Training data size X: ", x_test.shape, " Y train: ", y_test.shape)
class_names = ['airplane','automobile','bird','cat','deer',

               'dog','frog','horse','ship','truck']



fig = plt.figure(figsize=(8,3))

for i in range(num_classes):

    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])

    idx = np.where(y_train[:]==i)[0]

    features_idx = x_train[idx,::]

    img_num = np.random.randint(features_idx.shape[0])

    im = (features_idx[img_num,::])

    ax.set_title(class_names[i])

    plt.imshow(im)

plt.show()
y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)
model = Sequential()



model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))

model.add(Activation('relu'))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))



model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))



model.add(Dropout(0.5))

model.add(Dense(num_classes))

model.add(Activation('softmax'))
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='sgd', metrics=['accuracy'])



x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255

# ====> running the model with batch normalization and dropouts



# ===> Model without dropouts



model = Sequential()



model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))

model.add(Activation('relu'))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))



model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))



# model.add(Dropout(0.5))

model.add(Dense(num_classes))

model.add(Activation('softmax'))

model.summary()
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='sgd', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, shuffle=True)
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, shuffle=True)
# ===> Model without dropouts



model = Sequential()



model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))

model.add(Activation('relu'))

model.add(BatchNormalization())



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(BatchNormalization())



model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))



model.add(Dropout(0.5))

model.add(Dense(num_classes))

model.add(Activation('softmax'))

model.summary()
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='sgd', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, shuffle=True)
# ====> Use L2 regularization inplace of dropouts and see the result



# ===> Model without dropouts



model = Sequential()



model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))

model.add(Activation('relu'))

model.add(BatchNormalization())



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(BatchNormalization())



model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512,kernel_regularizer=l2(0.01)))

model.add(Activation('relu'))



model.add(Dropout(0.5))

model.add(Dense(num_classes))

model.add(Activation('softmax'))

model.summary()
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='sgd', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, shuffle=True)
# ===> Use dropouts with L2 regularization and batch normalization



# ====> Use L2 regularization inplace of dropouts and see the result



# ===> Model without dropouts



model = Sequential()



model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))

model.add(Activation('relu'))

model.add(BatchNormalization())



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(BatchNormalization())



model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512,kernel_regularizer=l2(0.01)))

model.add(Activation('relu'))



model.add(Dropout(0.5))

model.add(Dense(num_classes))

model.add(Activation('softmax'))

model.summary()
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='sgd', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, shuffle=True)
# =====> model with extra CNN layers

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',

                 input_shape=x_train.shape[1:]))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(128, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512,kernel_regularizer=l2(0.01)))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes))

model.add(Activation('softmax'))

model.summary()
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='sgd', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, shuffle=True)