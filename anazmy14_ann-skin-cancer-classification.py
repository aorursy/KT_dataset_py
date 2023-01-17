import numpy as np

import pandas as pd

import os

import cv2

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten , Conv2D , MaxPool2D

from keras import backend as K

from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
in_part1 = set (os.listdir(  "../input/skin-cancer-mnist-ham10000/ham10000_images_part_1" ))

df = pd.read_csv ( "../input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv" )
x_train = [] 

y_train = []

lesions = set()



for index , row in df.iterrows() :

    

    if row["lesion_id"] in lesions :  # removing duplicates 

        continue 

    lesions.add(row["lesion_id"]) 

    

    

    path = "../input/skin-cancer-mnist-ham10000/"

    if row["image_id"] + ".jpg" in in_part1 :  # checking if the image in part1 then load it    

        path += "ham10000_images_part_1/" + row["image_id"] + ".jpg" 

    else :                                     # if the image isn't in part 1 then load it from part 2 folder   

        path += "ham10000_images_part_2/" + row["image_id"] + ".jpg" 



    img = cv2.imread(path)  

    img = cv2.resize( img , (100 ,75))  

    x_train.append(img)  

    y_train.append(row["dx"])



y_train = pd.get_dummies(y_train)

y_train.head()
y_train = np.asarray(y_train)

mean = np.mean(x_train)

std = np.std(x_train)

x_train = (x_train - mean)/std

print (x_train.shape)

print (y_train.shape)

x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.1, random_state = 2 , stratify = y_train )
input_shape = (75, 100, 3)

num_classes = 7



model = Sequential()

model.add(Flatten(input_shape = input_shape))

model.add(Dense( 512 , activation = "relu" ))

model.add(Dense( 256 , activation = "relu" ))

model.add(Dense( 128 , activation = "relu" ))

model.add(Dense( 128 , activation = "relu" ))

model.add(Dense( 64 , activation = "relu" ))



model.add(Dense(num_classes, activation='softmax'))

model.summary()

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)  
datagen = ImageDataGenerator(

        rotation_range=10, 

        zoom_range = 0.1,  

        width_shift_range=0.1, 

        height_shift_range=0.1,   )  



datagen.fit(x_train)
epochs = 50 

batch_size = 128

history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_validate,y_validate),

                              verbose = 1, callbacks=[learning_rate_reduction] , shuffle = True )




