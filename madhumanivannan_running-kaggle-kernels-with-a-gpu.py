from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense



# Initialising the CNN

classifier = Sequential()



# Step 1 - Convolution

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Adding a second convolutional layer

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))



classifier.add(Conv2D(128, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Step 3 - Flattening

classifier.add(Flatten())



# Step 4 - Full connection

classifier.add(Dense(units = 500, activation = 'relu'))

classifier.add(Dense(units = 1, activation = 'sigmoid'))



# Compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
data_dir = "../input/cat-and-dog/training_set"

data_dir_1 = "../input/cat-and-dog/test_set"
from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory(data_dir,

                                                 target_size = (64, 64),

                                                 batch_size = 32,

                                                 class_mode = 'binary')



test_set = test_datagen.flow_from_directory(data_dir_1,

                                            target_size = (64, 64),

                                            batch_size = 32,

                                            class_mode = 'binary')
classifier.fit_generator(training_set,

                         steps_per_epoch = 8000,

                         epochs = 1,

                         validation_data = test_set,

                         validation_steps = 2000)
from keras.models import load_model

from keras.preprocessing import image

import numpy as np



test_image = image.load_img('../input/test-dog/dog_1.jpeg', target_size=(64,64))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis=0)
result=classifier.predict_classes(test_image)

result
# Imports for Deep Learning

from keras.layers import Conv2D, Dense, Dropout, Flatten

from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator



# ensure consistency across runs

from numpy.random import seed

seed(1)

from tensorflow import set_random_seed

set_random_seed(2)



# Imports to view data

import cv2

from glob import glob

from matplotlib import pyplot as plt

from numpy import floor

import random



def plot_three_samples(letter):

    print("Samples images for letter " + letter)

    base_path = '../input/asl_alphabet_train/asl_alphabet_train/'

    img_path = base_path + letter + '/**'

    path_contents = glob(img_path)

    

    plt.figure(figsize=(16,16))

    imgs = random.sample(path_contents, 3)

    plt.subplot(131)

    plt.imshow(cv2.imread(imgs[0]))

    plt.subplot(132)

    plt.imshow(cv2.imread(imgs[1]))

    plt.subplot(133)

    plt.imshow(cv2.imread(imgs[2]))

    return



plot_three_samples('A')
plot_three_samples('B')
data_dir = "../input/asl_alphabet_train/asl_alphabet_train"

target_size = (64, 64)

target_dims = (64, 64, 3) # add channel for RGB

n_classes = 29

val_frac = 0.1

batch_size = 64



data_augmentor = ImageDataGenerator(samplewise_center=True, 

                                    samplewise_std_normalization=True, 

                                    validation_split=val_frac)



train_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, shuffle=True, subset="training")

val_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, subset="validation")
my_model = Sequential()

my_model.add(Conv2D(64, kernel_size=4, strides=1, activation='relu', input_shape=target_dims))

my_model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))

my_model.add(Dropout(0.5))

my_model.add(Conv2D(128, kernel_size=4, strides=1, activation='relu'))

my_model.add(Conv2D(128, kernel_size=4, strides=2, activation='relu'))

my_model.add(Dropout(0.5))

my_model.add(Conv2D(256, kernel_size=4, strides=1, activation='relu'))

my_model.add(Conv2D(256, kernel_size=4, strides=2, activation='relu'))

my_model.add(Flatten())

my_model.add(Dropout(0.5))

my_model.add(Dense(512, activation='relu'))

my_model.add(Dense(n_classes, activation='softmax'))



my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
my_model.fit_generator(train_generator, epochs=5, validation_data=val_generator)