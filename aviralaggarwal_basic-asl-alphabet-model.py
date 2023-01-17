import cv2

from glob import glob

from datetime import datetime

from matplotlib import pyplot as plt

from numpy import floor

import random

import os
#visualizing the data

def data_sample(class_name,img_no=3):

    folder_path = '../input/asl_alphabet_train/asl_alphabet_train/'

    img_path = folder_path + class_name + '/**'

    path_contents = glob(img_path)

    

    plt.figure(figsize=(16,16))

    imgs = random.sample(path_contents, img_no)

    for i in range(img_no):

        plt.subplot(int('1'+str(img_no)+str(i+1)))

        plt.imshow(cv2.imread(imgs[i]))

    return
data_sample('A',4)
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D

from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator
data_dir = "../input/asl_alphabet_train/asl_alphabet_train"

target_size = (64, 64)

target_dims = (64, 64, 3) # add channel for RGB

n_classes = 29

val_frac = 0.25

batch_size = 64

#With a batch size of 32, the accuracy is lower and in some cases, completely diverged

# With 128 as the size, the accuracy is less than 10%, even after a much higher number of epochs



data_augmentor = ImageDataGenerator(samplewise_center=True, 

                                    samplewise_std_normalization=True, 

                                    validation_split=val_frac)



train_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, shuffle=True, subset="training")

val_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, subset="validation")
#noting down results for fixed epochs = 5



#results with large number of filters(64, ...512 in dense)

#validation set accuracy ~81%, ~12 mins training time

#NO SIGNIFICANT EFFECT OF DOUBLING THE FILTERS, ACCURACY REDUCES BY ~2%



#results of half the number of filters(32, ...256 in dense)

#without pool layers accuracy = (97.50, 80.11)      ~12 mins training time

            # maxes at lesser epochs = (~97, 84.5)

#Using a basic standard model with different layers



sample_model = Sequential()



# The input layer and the first layers 

sample_model.add(Conv2D(filters=32, kernel_size=4, strides=1, activation='relu', input_shape=target_dims))

sample_model.add(Conv2D(filters=32, kernel_size=4, strides=2, activation='relu'))

#sample_model.add(MaxPool2D(pool_size=(2,2)))

sample_model.add(Dropout(0.5))



sample_model.add(Conv2D(64, kernel_size=4, strides=1, activation='relu'))

sample_model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))

#sample_model.add(MaxPool2D(pool_size=(2,2)))

sample_model.add(Dropout(0.5))



#ADDING THESE TWO LAYERS INCREASED ACCURACY



sample_model.add(Conv2D(128, kernel_size=4, strides=1, activation='relu'))

sample_model.add(Conv2D(128, kernel_size=4, strides=2, activation='relu'))



sample_model.add(Flatten())

sample_model.add(Dropout(0.5))

sample_model.add(Dense(256, activation='relu'))

sample_model.add(Dense(n_classes, activation='softmax'))



sample_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
time_start = datetime.now()

sample_model.fit_generator(train_generator, epochs=10, validation_data=val_generator)

time_end = datetime.now()

print("time taken = ",(time_end-time_start))