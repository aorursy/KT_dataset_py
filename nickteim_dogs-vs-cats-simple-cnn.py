import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import keras

import cv2

import csv

import shutil

from glob import glob

from PIL import Image

from IPython.display import FileLink



import numpy as np 

import pandas as pd 



import os

print(os.listdir("../input"))

from glob import glob

import random

import cv2

import matplotlib.pylab as plt

from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, MaxPool2D, BatchNormalization, Input

from keras.models import Sequential

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from pathlib import Path

from keras.optimizers import Adam,RMSprop,SGD

print(os.listdir("../input/dogs-vs-cats-redux-kernels-edition/"))
!unzip ../input/dogs-vs-cats-redux-kernels-edition/train.zip

#!unzip ../input/dogs-vs-cats-redux-kernels-edition/test.zip
#make directories

os.mkdir('train/cats/')

os.mkdir('train/dogs/')
#filling the directori for cat images



cats = glob('train/cat*.jpg')

shuf = np.random.permutation(cats)



for i in range(len(cats)): 

    shutil.move(shuf[i], 'train/cats/')







#filling the directori for dog images    

dogs = glob('train/dog*.jpg')

shuf = np.random.permutation(dogs)



for i in range(len(dogs)): 

    shutil.move(shuf[i], 'train/dogs/')



image_path = 'train/'

images_dict = {}





for image in os.listdir(image_path):

    folder_path = os.path.join(image_path, image)

    images = os.listdir(folder_path)

    

    images_dict[image] = [folder_path, image]

    img_idx = random.randint(0,len(image)-1)

    image_img_path = os.path.join(image_path, image, images[img_idx])

    #printing image

    img = cv2.imread(image_img_path)

    #print(image_img_path) # to get the path of one image with the .jpg number; uncommen this line

    plt.imshow(img);
height=150

width=150

batch_size=32     

seed=1337



train_dir = Path('train/')

test_dir = Path('train/')



# Training generator first step rescale and gives more images in different angels and zoom range and even flipping the image

train_datagen = ImageDataGenerator(rotation_range = 30       

                                   ,rescale=1. / 255,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   horizontal_flip=True, validation_split = 0.2) #splitting data in traininf



train_generator = train_datagen.flow_from_directory(train_dir, #load data 

                                                    target_size=(height,width), #what size image we want

                                                    batch_size=batch_size,  #how many images to read at the time 

                                                    seed=seed,

                                                    class_mode='categorical', #we are classifing images into different categories

                                                    subset = "training")      # we use the subset created for training data





# Test generator we do the same as in train_generator without the rotation on images. 

test_datagen = ImageDataGenerator(rescale=1./255, validation_split = 0.2) #splitting data in validation

test_generator = test_datagen.flow_from_directory(test_dir, 

                                                  target_size=(height,width), 

                                                  batch_size=batch_size,

                                                  seed=seed,

                                                  class_mode='categorical',

                                                  subset = "validation")
model = Sequential()

model.add(Conv2D(12, kernel_size=(3,3),

                 activation='relu',

                 input_shape=(150, 150, 3)))
model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.5))                    #dropout means how offent to drop nodes, to make new connections

model.add(Dense(256,activation='relu'))    #relu means rectified linear unit and is y=max(0, x) and 'Dense' means how dense you want the model in the given activation

model.add(Dense(2, activation='softmax')) #softmax turns it into properbelities
model.compile(optimizer = 'adam',loss="categorical_crossentropy", metrics=["accuracy"])   #you can change (optimizer = 'adam') to (Adam(lr=0.0001)) here is lr=learning rate

# compile tells tenserflow how to update the dense connections, when we are training on the data

fitting_model = model.fit_generator(train_generator,

                    steps_per_epoch = 1097//batch_size, #just a calulation (train size/batch size) also how many pictures we want to load each time

                    validation_data = test_generator, 

                    validation_steps = 272//batch_size, #just a calulation (validation size/batch size)

                    epochs = 2,                       #epochs means how many cycle we want the model to go though our dataset

                    verbose  = 1)                     #verbose just means whar you want to see while the model is training, 0=nothing, 1=a bar of proces, 2=the number of runs it wil take