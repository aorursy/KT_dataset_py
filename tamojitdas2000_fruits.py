import os

import cv2

import matplotlib.pyplot as plt

import numpy as np

import tensorflow as tf

import random

    
path='/kaggle/input/fruits/fruits-360/'

IMG_SIZE = 224

BATCH_SIZE = 32
traindatagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.0,

                                                              shear_range = 0.2,

                                                               zoom_range = 0.2,

                                                               height_shift_range = 0.2,

                                                               width_shift_range = 0.2,

                                                               fill_mode = "nearest"                                                                                                       

                                                              )

train_data = traindatagen.flow_from_directory(os.path.join(path,"Training"), target_size=(IMG_SIZE,IMG_SIZE),batch_size=BATCH_SIZE,class_mode="categorical")
testdatagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.0)                                                                                                                                                        

                                                              

test_data = testdatagen.flow_from_directory(os.path.join(path,"Test"), target_size=(IMG_SIZE,IMG_SIZE),batch_size=BATCH_SIZE,class_mode="categorical")
categories = list(train_data.class_indices.keys())

print(categories)
train_data.image_shape
class myCallBack(tf.keras.callbacks.Callback):

    def on_epoch_end(self,epoch,logs={}):

        if(logs.get('accuracy') > 0.95):

            self.model.stop_training = True



callback=myCallBack()





model=tf.keras.models.Sequential()



model.add(tf.keras.layers.Conv2D(64,(3,3), input_shape=(IMG_SIZE,IMG_SIZE,3),activation='relu'))

model.add(tf.keras.layers.MaxPooling2D(2,2))



model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))

model.add(tf.keras.layers.MaxPooling2D(2,2))



model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))

model.add(tf.keras.layers.MaxPooling2D(2,2))



model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))

model.add(tf.keras.layers.MaxPooling2D(2,2))



model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))

model.add(tf.keras.layers.MaxPooling2D(2,2))



model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)))

#model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Dense(128,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)))

#model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Dense(128,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)))

model.add(tf.keras.layers.Dense(len(categories),activation='softmax'))



model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit_generator(train_data, validation_data=test_data, epochs=1)#, callbacks=[callback])
model.summary()