# Necessary Imports
import numpy as np
import os

import random

import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from keras.preprocessing.image import ImageDataGenerator,load_img 


import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import MaxPool2D
from keras.layers import MaxPooling2D


%matplotlib inline
os.listdir('../input/chest-xray-pneumonia/chest_xray/')
homeDirectory="../input/chest-xray-pneumonia/chest_xray/"
trainDirectory = '../input/chest-xray-pneumonia/chest_xray/train/'
valDirectory = '../input/chest-xray-pneumonia/chest_xray/val/'
testDirectory = '../input/chest-xray-pneumonia/chest_xray/test/'
os.listdir('../input/chest-xray-pneumonia/chest_xray/train/'+'NORMAL')
# Peeking at the our train images
randomChoiceInt=random.randint(0,len(os.listdir(trainDirectory+'NORMAL')))

normalSampleImage=Image.open(trainDirectory+'NORMAL/'+os.listdir(trainDirectory+'NORMAL/')[randomChoiceInt])
pneumonialSampleImage=Image.open(trainDirectory+'PNEUMONIA/'+os.listdir(trainDirectory+'PNEUMONIA/')[randomChoiceInt])


fig = plt.figure(figsize=(20,16))
normPlt=fig.add_subplot(121)
plt.imshow(normalSampleImage,cmap='gray')
normPlt.set_title("Normal X-RAY")


pnePlot=fig.add_subplot(122)
plt.imshow(pneumonialSampleImage,cmap='gray')
pnePlot.set_title("Pnemonial X-RAY")

plt.show()
pneumonialSampleImage.width,pneumonialSampleImage.height
model=Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(activation='relu',units=129))
model.add(Dense(activation='relu',units=1))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()
trainDataGenerator=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

valDataGenerator=ImageDataGenerator(rescale=1./255)

trainingImageSet=trainDataGenerator.flow_from_directory(trainDirectory,batch_size=32,class_mode='binary',target_size=(32,32))
testImageSet=valDataGenerator.flow_from_directory(testDirectory,batch_size=32,class_mode='binary',target_size=(32,32))
valImageSet=valDataGenerator.flow_from_directory(valDirectory,batch_size=32,class_mode='binary',target_size=(32,32))
model.fit_generator(trainingImageSet,epochs=30,
                    validation_data=valImageSet,
                    steps_per_epoch=15,
                    validation_steps=624
                   )
