# Importing all libraries



import numpy as np

import cv2

import matplotlib.pyplot as plt

import os

import glob

import random

import tensorflow as tf

from tensorflow import keras

from keras.layers import Conv2D, Dense, Flatten, Activation, Dropout, MaxPooling2D

from keras.models import Sequential



from keras.applications import mobilenet, VGG19



# location of images

DATADIR = '../input/flowers-recognition/flowers/flowers'

CATEGORIES = os.listdir('../input/flowers-recognition/flowers/flowers')

print(CATEGORIES)
# Importing all images as 3D arrays, resizing them and appending to form a list of image arrays along with label names



training_data = []



img_size = 128



def create_training_data():

    for category in CATEGORIES:

        path = os.path.join(DATADIR, category)

        class_num = CATEGORIES.index(category)

        # class_num = category

        for img in os.listdir(path):

            try: 

                img_array = cv2.imread(os.path.join(path, img))

                new_array = cv2.resize(img_array,(img_size, img_size))

                training_data.append([new_array, class_num])

            except Exception as e:

                pass



create_training_data()



#shuffling the data

random.shuffle(training_data)
len(training_data)
# pre-processing the dataset

X = []

Y = []

for features, label in training_data:

    X.append(features)

    Y.append(label)



from keras.utils import to_categorical

#one-hot encoding of labels

Y=to_categorical(Y,5)



#normalisation of pixel intensities

X=np.array(X)/255.0



# CNN model built from scratch 





num_classes = 5

model = Sequential()





model.add(Conv2D(64,kernel_size=(3,3), input_shape= X.shape[1:]))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(64,kernel_size=(3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Flatten())

model.add(Dense(64,activation='relu'))





model.add(Dense(num_classes))

model.add(Activation('softmax'))





model.compile(loss = 'categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])

model.fit(X, Y, batch_size = 64, epochs=10, validation_split=0.1)

# model.fit(x_train,y_train,epochs=10,batch_size=100)

# Using weights from MobileNet model



my_new_model = Sequential()

my_new_model.add(mobilenet.MobileNet(include_top=False, pooling='avg', weights='imagenet'))

my_new_model.add(Dense(num_classes, activation='softmax'))



# Say not to train first layer (ResNet) model. It is already trained

my_new_model.layers[0].trainable = False



my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

my_new_model.fit(X, Y, batch_size= 64, epochs=10, validation_split=0.1)
my_new_model = Sequential()

my_new_model.add(VGG19(include_top=False, pooling='avg', weights='imagenet'))

my_new_model.add(Dense(num_classes, activation='softmax'))



# Say not to train first layer (ResNet) model. It is already trained

my_new_model.layers[0].trainable = False



my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

my_new_model.fit(X, Y,  epochs=10, validation_split=0.1)