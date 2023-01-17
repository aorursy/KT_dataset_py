import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as ts

import cv2

import os

from PIL import Image

import glob

from sklearn.utils import shuffle

# Keras Imports



from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,Input,Activation,ZeroPadding2D

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.losses import binary_crossentropy





# Sci-kit learn imports

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split



from keras.initializers import glorot_uniform

from keras.layers import Dropout , Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,MaxPool2D

from keras.models import Model, load_model

from keras.initializers import glorot_uniform

from sklearn.model_selection import train_test_split

import keras.backend as K

from sklearn.utils import shuffle



Uninfected = glob.glob(r'C:\Users\G.VENKATARAMANA\DataSets\cell-images-for-detecting-malaria\cell_images\cell_images\Uninfected\*.png')

Infected = glob.glob(r'C:\Users\G.VENKATARAMANA\DataSets\cell-images-for-detecting-malaria\cell_images\cell_images\Parasitized\*.png')

img_arr = list() 

label = list()



for i in Infected:

    img =cv2.imread(i)

    img_res = cv2.resize(img,(64,64))

    img_arr.append(img_res)

    label.append(2)

data = []          #EMPTY LIST

labels = []        #EMPTY LIST



for i in Infected:

    img = cv2.imread(i)

    resize_img =cv2.resize(img,(50,50))  #RESIZING 

    data.append(np.array(resize_img))   #COMBINING WITH A EMPTY LIST

    labels.append(0)                    #labelling



for i in Uninfected:

    img = cv2.imread(i)

    resize_img = cv2.resize(img,(50,50)) #RESIZING 

    data.append(np.array(resize_img))    #COMBINING WITH A EMPTY LIST

    labels.append(1)                      #labelling
print(len(img))        #NUMBER OF IMAGE
img_arr  = np.array(data)    #converting in to an ARRAY

label =np.array(labels)

img_arr.shape , label.shape   #IMAGE SHAPE AND LABEL SIZE
img_arr,label = shuffle(img_arr, label,random_state = 0)        



X_train,X_test, y_train,y_test = train_test_split(img_arr ,label,test_size = 0.2,random_state = 0)  #SPLITTING THE DATA INTO TRAINING AND TESTING.
type(y_train)
classifier = Sequential()



#Adding 1st Convolution and Pooling Layer

classifier.add(Conv2D(32,kernel_size=(3,3),input_shape = (64, 64, 3), activation = 'relu'))

classifier.add(MaxPool2D(pool_size=(2,2)))



#Adding 2nd Convolution and Pooling Layer

classifier.add(Conv2D(32,kernel_size=(3,3),activation = 'relu'))

classifier.add(MaxPool2D(pool_size=(2,2)))



#Adding 3rd Convolution and Pooling Layer

classifier.add(Conv2D(32,kernel_size=(3,3), activation = 'relu'))

classifier.add(MaxPool2D(pool_size=(2,2)))



#Adding 4th Convolution and Pooling Layer

classifier.add(Conv2D(32,kernel_size=(3,3), activation = 'relu'))

classifier.add(MaxPool2D(pool_size=(2,2)))



#Flattening

classifier.add(Flatten())



classifier.add(Dense(units=128,activation='relu'))

classifier.add(Dense(units = 1, activation = 'sigmoid'))



classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

parasite_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

uninfected_datagen = ImageDataGenerator(rescale=1./255)

parasite_data = parasite_datagen.flow_from_directory(r'C:\Users\G.VENKATARAMANA\DataSets\cell-images-for-detecting-malaria\cell_images\cell_images',

                                                     target_size=(64,64),

                                                     batch_size=32,

                                                     class_mode = 'binary')

uninfected_data = uninfected_datagen.flow_from_directory(r'C:\Users\G.VENKATARAMANA\DataSets\cell-images-for-detecting-malaria\cell_images\cell_images',

                                                        target_size=(64,64),

                                                        batch_size=32,

                                                        class_mode = 'binary')
accuracies = classifier.fit_generator(parasite_data,

                         steps_per_epoch = 100,

                         epochs = 30,

                         validation_data = uninfected_data,

                         validation_steps = 50)
acc = pd.DataFrame.from_dict(accuracies.history)

acc = pd.concat([pd.Series(range(0,30),name='epochs'),acc],axis=1)

acc.head(30)



