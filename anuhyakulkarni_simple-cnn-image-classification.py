#import basic libraries
import numpy as np
import pandas as pd
import keras
#import all the essential packages from keras , required for CNN

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten





classifier = Sequential()

classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation = 'relu'))  # 32 is the number of feature detectors, and 3*3 window

classifier.add(MaxPooling2D(pool_size=(2,2)))    #size of the sub table ,min is considered = 2*2 

classifier.add(Flatten())       # flattens the sub table into a linear to be able to fed into training
classifier.add(Dense(128,activation = 'relu'))      # first and input layer
classifier.add(Dense(6,activation = 'softmax'))    #output layer, 6 is the no of output categories 
#compile the model

classifier.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#sparse categorical bcz the output has multiple catgories
#image data preprocessing 

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2, #shaering transformation
        zoom_range=0.2,       # zoom in required
        horizontal_flip=True) #if the images data needs to be horizontally flipped, applicable for real world images

test_datagen = ImageDataGenerator(rescale=1./255) #rescale the image if necessary (RGB coefficients normalize)


#creates the train set

train_set = train_datagen.flow_from_directory("../input/intel-image-classification/seg_train/seg_train",
        target_size=(64,64), #size of the image in the model 
        batch_size=32,
        class_mode='sparse')

#creates the test set
test_set = test_datagen.flow_from_directory(
        '../input/intel-image-classification/seg_test/seg_test',
        target_size=(64,64),       #size of the image in the model
        batch_size=32,
        class_mode='binary')
#fit the model

classifier.fit_generator(train_set,
                    steps_per_epoch=14034,     #number of images
                    epochs=5,
                    validation_data=test_set,
                     validation_steps=3000)