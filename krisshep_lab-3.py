# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import keras

from keras.datasets import fashion_mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

import matplotlib.pyplot as plt
batch_size = 128

num_classes = 10

epochs = 10

img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols)

test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)



train_labels = keras.utils.to_categorical(train_labels, num_classes)

test_labels = keras.utils.to_categorical(test_labels, num_classes)



train_images = train_images/255.0

test_images = test_images/255.0

plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.imshow(train_images[i])

plt.show()

train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)
model = Sequential()

model.add(Conv2D(32, kernel_size = (3,3),

                activation = 'relu',

                input_shape = input_shape))

model.add(Conv2D(64, kernel_size = (3,3),

                activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation = 'relu'))

model.add(Dense(num_classes, activation = 'softmax'))
model.compile(loss= keras.losses.categorical_crossentropy,

             optimizer = keras.optimizers.Adam(),

             metrics=['accuracy'])
model.fit(train_images, train_labels,

         batch_size = batch_size, 

         epochs = epochs,

         verbose = 1)
score = model.evaluate(test_images, test_labels, verbose=0)

print('Test loss: ', score[0])

print('Test accuracy: ', score[1])
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
batch_size_0 = 128

num_classes_0 = 10

epochs_0 = 10

img_rows_0, img_cols_0 = 32, 32

input_shape_0 = (img_rows_0, img_cols_0, 3)
y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)
plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.imshow(x_train[i])

plt.show()
x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255
val_part = 0.15
model_0 = Sequential()

model_0.add(Conv2D(32, kernel_size = (3,3),

                activation = 'relu',

                input_shape = input_shape_0))

model_0.add(Conv2D(64, kernel_size = (3,3),

                activation = 'relu'))

model_0.add(MaxPooling2D(pool_size=(2,2)))

model_0.add(Flatten())

model_0.add(Dense(128, activation = 'relu'))

model_0.add(Dense(num_classes, activation = 'softmax'))



model_0.compile(loss= keras.losses.categorical_crossentropy,

             optimizer = keras.optimizers.Adam(),

             metrics=['accuracy'])



hist_0 = model_0.fit(x_train, y_train,

         batch_size = batch_size_0, 

         epochs = epochs_0,

         verbose = 1,

         validation_split=val_part)
score_0 = model_0.evaluate(x_test, y_test, verbose=0)

print('Test loss: ', score_0[0])

print('Test accuracy: ', score_0[1])
def plot_fit_res(hist):

    plt.figure(figsize=plt.figaspect(0.5))

    plt.subplot(1, 2, 1)

    l = range(0, len(hist.history['val_loss']))

    plt.plot(l, hist.history['val_loss'])

    plt.title('val_loss')

    plt.subplot(1, 2, 2)

    l = range(0, len(hist.history['val_accuracy']))

    plt.plot(l, hist.history['val_accuracy'])

    plt.title('val_accuracy')
plot_fit_res(hist_0)
epochs_1 = 20
model_1 = Sequential()



model_1.add(Conv2D(32, kernel_size = (3, 3),

           input_shape = input_shape_0,

           activation = 'relu'))

model_1.add(Conv2D(32, kernel_size = (3,3),

           activation = 'relu'))

model_1.add(MaxPooling2D(pool_size = (2,2)))



model_1.add(Conv2D(64, kernel_size = (3, 3),

           input_shape = input_shape_0,

           activation = 'relu'))

model_1.add(Conv2D(64, kernel_size = (3,3),

           activation = 'relu'))

model_1.add(MaxPooling2D(pool_size = (2,2)))

            

model_1.add(Flatten())



model_1.add(Dense(256, activation = 'relu'))

model_1.add(Dense(num_classes_0, activation = 'softmax'))

model_1.compile(loss = keras.losses.categorical_crossentropy,

               optimizer = keras.optimizers.Adam(),

               metrics=['accuracy'])



hist_1 = model_1.fit(x_train, y_train,

           batch_size = batch_size_0,

           epochs = epochs_1,

           verbose = 1,

           validation_split=val_part)
score_1 = model_1.evaluate(x_test, y_test, verbose = 0)

print('Test loss: ', score_1[0])

print('Test accuracy: ', score_1[1])

plot_fit_res(hist_1)
dp_koef = 0.1



model_2 = Sequential()



model_2.add(Conv2D(32, kernel_size = (3, 3),

           input_shape = input_shape_0,

           activation = 'relu'))

model_2.add(Conv2D(32, kernel_size = (3,3),

           activation = 'relu'))

model_2.add(MaxPooling2D(pool_size = (2,2)))

model_2.add(Dropout(dp_koef))



model_2.add(Conv2D(64, kernel_size = (3, 3),

           input_shape = input_shape_0,

           activation = 'relu'))

model_2.add(Conv2D(64, kernel_size = (3,3),

           activation = 'relu'))

model_2.add(MaxPooling2D(pool_size = (2,2)))

model_2.add(Dropout(dp_koef))

            

model_2.add(Flatten())



model_2.add(Dense(256, activation = 'relu'))

model_2.add(Dropout(dp_koef*2))

model_2.add(Dense(num_classes_0, activation = 'softmax'))



model_2.compile(loss= keras.losses.categorical_crossentropy,

             optimizer = keras.optimizers.Adam(),

             metrics=['accuracy'])



hist_2 = model_2.fit(x_train, y_train,

         batch_size = batch_size_0, 

         epochs = epochs_1,

         verbose = 1,

         validation_split = val_part)
score_2 = model_2.evaluate(x_test, y_test, verbose=0)

print('Test loss: ', score_2[0])

print('Test accuracy: ', score_2[1])

plot_fit_res(hist_2)
from keras.layers import Activation

from keras.layers import AveragePooling2D, Input

from keras.models import Model

from keras.regularizers import l2
dp_koef = 0.1

def resnet_layer(inputs,

                 num_filters=16,

                 kernel_size=3,

                 strides=1,

                 activation='relu',

                 dropout=True,

                 conv_first=True):

    

    conv = Conv2D(num_filters,

                  kernel_size=kernel_size,

                  strides=strides,

                  padding='same',

                  kernel_initializer='he_normal',

                  kernel_regularizer=l2(1e-4))



    x = inputs

    if conv_first:

        x = conv(x)

        if activation is not None:

            x = Activation(activation)(x)

        if dropout:

            x = Dropout(dp_koef)(x)

    else:

        if activation is not None:

            x = Activation(activation)(x)

        if dropout:

            x = Dropout(dp_koef)(x)

        x = conv(x)

    return x





def resnet_v1(input_shape, depth, num_classes=10):

    if (depth - 2) % 6 != 0:

        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')

    # Start model definition.

    num_filters = 16

    num_res_blocks = int((depth - 2) / 6)



    inputs = keras.engine.input_layer.Input(shape=input_shape)

    x = resnet_layer(inputs=inputs)

    # Instantiate the stack of residual units

    for stack in range(3):

        for res_block in range(num_res_blocks):

            strides = 1

            if stack > 0 and res_block == 0:  # first layer but not first stack

                strides = 2  # downsample

            y = resnet_layer(inputs=x,

                             num_filters=num_filters,

                             strides=strides, dropout=True)

            y = resnet_layer(inputs=y,

                             num_filters=num_filters,

                             activation=None, dropout=True)

            if stack > 0 and res_block == 0:  # first layer but not first stack

                # linear projection residual shortcut connection to match

                # changed dims

                x = resnet_layer(inputs=x,

                                 num_filters=num_filters,

                                 kernel_size=1,

                                 strides=strides,

                                 activation=None,

                                 dropout=True)

            x = keras.layers.add([x, y])

            x = Activation('relu')(x)

        num_filters *= 2



    # Add classifier on top.

    # v1 does not use BN after last shortcut connection-ReLU

    x = AveragePooling2D(pool_size=8)(x)

    y = Flatten()(x)

    outputs = Dense(num_classes,

                    activation='softmax',

                    kernel_initializer='he_normal')(y)



    # Instantiate model.

    model = Model(inputs=inputs, outputs=outputs)

    return model

depth = 3*6+2 #from cifar10 example

model_3 = resnet_v1(input_shape=input_shape_0, depth=depth)



model_3.compile(loss= keras.losses.categorical_crossentropy,

             optimizer = keras.optimizers.Adam(),

             metrics=['accuracy'])



hist_3 = model_3.fit(x_train, y_train,

         batch_size = batch_size_0, 

         epochs = 40,

         verbose = 1,

         validation_split = val_part)
score_3 = model_3.evaluate(x_test, y_test, verbose=0)

print('Test loss: ', score_3[0])

print('Test accuracy: ', score_3[1])

plot_fit_res(hist_3)
import tensorflow as tf



y_pred = model_1.predict_classes(x_test)

y_test_cm = [np.argmax(y_test[i]) for i in range(0,y_test.shape[0])]

#print(y_test[:10])

#print(y_test_cm[:10])

conf_mat = tf.math.confusion_matrix(labels = y_test_cm, predictions = y_pred).numpy()

print(conf_mat)
conf_mat_nm = np.around(conf_mat.astype('float')/conf_mat.sum(axis = 1)[:np.newaxis], decimals = 2)

print(conf_mat_nm)
[print(i,': ', conf_mat_nm[i,i]) for i in range(0, num_classes_0)]