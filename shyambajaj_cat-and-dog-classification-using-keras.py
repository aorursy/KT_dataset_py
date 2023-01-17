#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Wed Feb 13 16:14:50 2019



@author: shyam

"""



from keras.models import Sequential

from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout

import tensorflow as tf

import keras

import os

import cv2



config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 

sess = tf.Session(config=config) 

keras.backend.set_session(sess)

S=64



#From Keras Documentation

from keras.preprocessing.image import ImageDataGenerator



trainDatagen = ImageDataGenerator(

                    rescale=1./255,

                    shear_range=0.2,

                    zoom_range=0.2,

                    horizontal_flip=True)



testDatagen = ImageDataGenerator(rescale=1./255)



trainDataset = trainDatagen.flow_from_directory(

        '../input/training_set/training_set',

        target_size=(S, S),

        batch_size=32,

        class_mode='binary')



testDataset = testDatagen.flow_from_directory(

        '../input/test_set/test_set',

        target_size=(S, S),

        batch_size=32,

        class_mode='binary')
#Initializing CNN

classifier = Sequential()
#Adding 1st Convolution Layer

classifier.add(Convolution2D(filters=32, kernel_size=(3,3), strides=(1,1), input_shape=(S,S,3), activation='relu', padding='same'))



#Adding 1st MaxPooling Layer to reduce the size of feature map

classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2) ))



#Adding 1st BatchNormalization Layer for higher Learning Rate

classifier.add(BatchNormalization())



#Adding 1st Dropout Layer to eliminate overfitting

#classifier.add(Dropout(0.2))
#Adding 2nd Convolution Layer

classifier.add(Convolution2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'))



#Adding 2nd MaxPooling Layer to reduce the size of feature map

classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



#Adding 2nd BatchNormalization Layer for higher Learning Rate

classifier.add(BatchNormalization())



#Adding 2nd Dropout Layer to eliminate overfitting

classifier.add(Dropout(0.2))
"""

#Adding 3rd Convolution Layer

classifier.add(Convolution2D(filters=32, kernel_size=(3,3), strides=(2,2), activation='relu', padding='same'))



#Adding 3rd MaxPooling Layer to reduce the size of feature map

classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



#Adding 3rd BatchNormalization Layer for higher Learning Rate

classifier.add(BatchNormalization())



#Adding 3rd Dropout Layer to eliminate overfitting

#classifier.add(Dropout(0.2))

"""
#Adding Flatten Layer to convert 2D matrix into an array

classifier.add(Flatten())
#Adding Fully connected layer

classifier.add(Dense(units=32,activation='relu'))



#Adding Output Layer

classifier.add(Dense(units=1,activation='sigmoid'))

print(classifier.summary())
#Compiling the CNN

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



#Fitting the CNN to images

history = classifier.fit_generator(trainDataset,

                         steps_per_epoch=8005,

                         epochs=10,

                         validation_data=testDataset,

                         validation_steps=2000,

                         verbose = 1)
from matplotlib import pyplot as plt

plt.plot(history.history['acc'],'green',label='Accuracy')

plt.plot(history.history['loss'],'red',label='Loss')

plt.title('Training Accuracy & Loss')

plt.xlabel('Epoch')

plt.figure()

plt.plot(history.history['val_acc'],'green',label='Accuracy')

plt.plot(history.history['val_loss'],'red',label='Loss')

plt.title('Validation Accuracy & Loss')

plt.xlabel('Epoch')

plt.figure()
from matplotlib import pyplot as plt



directory = os.listdir("../input/test_set/test_set/cats")

print(directory[10])



imgCat = cv2.imread("../input/test_set/test_set/cats/" + directory[10])

plt.imshow(imgCat)



imgCat = cv2.resize(imgCat, (S,S))

imgCat = imgCat.reshape(1,S,S,3)



pred = classifier.predict(imgCat)

print("Probability that it is a Cat = ", "%.2f" % (1-pred))



directory = os.listdir("../input/test_set/test_set/dogs" )

print(directory[10])



imgDog = cv2.imread("../input/test_set/test_set/dogs/" + directory[10])

plt.imshow(imgDog)



imgDog = cv2.resize(imgDog, (S,S))

imgDog = imgDog.reshape(1,S,S,3)



pred = classifier.predict(imgDog)

print("Probability that it is a Dog = ", "%.2f" % pred)