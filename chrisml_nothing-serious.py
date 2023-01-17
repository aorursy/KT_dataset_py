from __future__ import print_function

import numpy as np

np.random.seed(1337)  # for reproducibility

import pandas as pd



from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D

from keras.utils import np_utils

from keras import backend as K



import tensorflow as tf



tf.python.control_flow_ops = tf



from PIL import Image

def read_labeled_image_list(image_list_file):

    csv = pd.read_csv("../input/"+image_list_file)

    filenames  = ["../input/images/" + str(x) + ".jpg" for x in csv['id']]

    labels = csv['species']

    return filenames, labels



image_list, y_train = read_labeled_image_list("train.csv")




batch_size = 28

nb_classes = 100

nb_epoch = 12



# input image dimensions

img_rows, img_cols = 100, 100

# number of convolutional filters to use

nb_filters = 64

# size of pooling area for max pooling

pool_size = (2, 2)

# convolution kernel size

kernel_size = (3, 3)





def read_labeled_image_list(image_list_file):

    csv = pd.read_csv("../input/"+image_list_file)

    filenames  = ["../input/images/" + str(x) + ".jpg" for x in csv['id']]

    labels = csv['species']

    return filenames, labels



def read_image_list(image_list_file):

    csv = pd.read_csv("../input/"+image_list_file)

    filenames  = ["../input/images/" + str(x) + ".jpg" for x in csv['id']]

    return filenames



image_list, y_train = read_labeled_image_list("train.csv")

X_train = []





for i in image_list:

    try:

        img = Image.open(i)

        X_train.append(np.array(img.getdata(),np.uint8).reshape(img.size[1], img.size[0], 1))

    except:

        print("Error train image: " + i)

X_train=np.array(X_train)





image_list_test = read_image_list("test.csv")



X_test = []



for i in image_list_test:

    try:

        img = Image.open(i)

        X_test.append(np.array(img.getdata(),np.uint8).reshape(img.size[1], img.size[0], 1))

    except Exception as e:

        print(e)

        print("Error test image: '"+i+"'")

X_test=np.array(X_test)



if K.image_dim_ordering() == 'th':

    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)

    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

    input_shape = (1, img_rows, img_cols)

else:

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)

    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

    input_shape = (img_rows, img_cols, 1)



X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train /= 255

X_test /= 255

print('X_train shape:', X_train.shape)

print(X_train.shape[0], 'train samples')

print(X_test.shape[0], 'test samples')



# convert class vectors to binary class matrices

Y_train = np_utils.to_categorical(y_train, nb_classes)

Y_test = np_utils.to_categorical(y_test, nb_classes)



model = Sequential()



model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],

                        border_mode='valid',

                        input_shape=input_shape))

model.add(Activation('relu'))

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=pool_size))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(nb_classes))

model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy',

              optimizer='adadelta',

              metrics=['accuracy'])



model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,

          verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])

print('Test accuracy:', score[1])