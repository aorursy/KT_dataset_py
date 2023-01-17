
import glob
import shutil
import os

input_dir = "../input/train/"
dest_dir = "../input/"
train_cate = 'train/cate/'
train_doge = 'train/doge/'
valid_cate = 'validate/cate/'
valid_doge = 'validate/doge/'
    
if not os.path.exists(train_cate):
    os.makedirs(train_cate)
    os.makedirs(train_doge)
    
if not os.path.exists(valid_cate):
    os.makedirs(valid_cate)
    os.makedirs(valid_doge)

for file in os.listdir("../input/train/"):
    if "dog" in file:
        shutil.copyfile(input_dir+file, train_doge+file)
    if "cat" in file:
        shutil.copyfile(input_dir+file, train_cate+file)

print('files copied!')
        
doge_Files = os.listdir(train_doge)
for file in doge_Files[0:2000]:
    shutil.move(train_doge+file, valid_doge+file)
    
cate_Files = os.listdir(train_cate)
for file in cate_Files[0:2000]:
    shutil.move(train_cate+file, valid_cate+file)
    
print('Validation files separated!')

import tensorflow as tf
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

from keras.backend.tensorflow_backend import set_session  
config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  
set_session(tf.Session(config=config)) 
# dimensions of our images.
img_width, img_height = 450, 450

#train_data_dir = 'D:/NeuralNetworkData/MammographyImges/train/'
#validation_data_dir = 'D:/NeuralNetworkData/MammographyImges/validate/'

train_data_dir = 'train'
validation_data_dir = 'validate'
epochs = 5
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
#https://towardsdatascience.com/keras-a-thing-you-should-know-about-keras-if-you-plan-to-train-a-deep-learning-model-on-a-large-fdd63ce66bd2
#https://medium.com/@parthvadhadiya424/hello-world-program-in-keras-with-cnn-dog-vs-cat-classification-efc6f0da3cc5
#https://towardsdatascience.com/build-your-first-deep-learning-classifier-using-tensorflow-dog-breed-example-964ed0689430

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='',
                                            histogram_freq=0,
                                            write_graph=True,
                                            write_images=True)

history = model.fit_generator(
            train_generator,
            steps_per_epoch=250,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=250,
            callbacks=[tbCallBack])

model.save('doggo_cate.h5')

