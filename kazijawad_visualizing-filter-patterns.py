from os import listdir

from os.path import isfile, join



mypath = "../input/datasets/images/"

file_names = [f for f in listdir(mypath) if isfile(join(mypath, f))]

print(str(len(file_names)) + " images loaded")
import cv2 as cv

import numpy as np

import sys

import os

import shutil



dog_count = 0

cat_count = 0

training_size = 1000

test_size = 500

training_images = []

training_labels = []

test_images = []

test_labels = []

size = 150

dog_dir_train = "./datasets/catsvsdogs/train/dogs/"

cat_dir_train = "./datasets/catsvsdogs/train/cats/"

dog_dir_val = "./datasets/catsvsdogs/validation/dogs/"

cat_dir_val = "./datasets/catsvsdogs/validation/cats/"



def make_dir(directory):

    if os.path.exists(directory):

        shutil.rmtree(directory)

    os.makedirs(directory)

    

make_dir(dog_dir_train)

make_dir(cat_dir_train)

make_dir(dog_dir_val)

make_dir(cat_dir_val)



def getZeros(number):

    if number > 10 and number < 100:

        return "0"

    elif number < 10:

        return "00"

    else:

        return ""

    

for i, file in enumerate(file_names):

    if file_names[i][0] == "d":

        dog_count += 1

        image = cv.imread(mypath + file)

        image = cv.resize(image, (size, size), interpolation=cv.INTER_AREA)

        

        if dog_count <= training_size:

            training_images.append(image)

            training_labels.append(1)

            zeros = getZeros(dog_count)

            cv.imwrite(dog_dir_train + "dog" + str(zeros) + str(dog_count) + ".jpg", image)

        elif dog_count > training_size and dog_count <= training_size + test_size:

            test_images.append(image)

            test_labels.append(1)

            zeros = getZeros(dog_count - 1000)

            cv.imwrite(dog_dir_val + "dog" + str(zeros) + str(dog_count - 1000) + ".jpg", image)      

    elif file_names[i][0] == "c":

        cat_count += 1

        image = cv.imread(mypath + file)

        image = cv.resize(image, (size, size), interpolation=cv.INTER_AREA)

        

        if cat_count <= training_size:

            training_images.append(image)

            training_labels.append(0)

            zeros = getZeros(cat_count)

            cv.imwrite(cat_dir_train + "cat" + str(zeros) + str(cat_count) + ".jpg", image)

        elif cat_count > training_size and cat_count <= training_size + test_size:

            test_images.append(image)

            test_labels.append(0)

            zeros = getZeros(cat_count - 1000)

            cv.imwrite(cat_dir_val + "cat" + str(zeros) + str(cat_count - 1000) + ".jpg", image)

            

    if dog_count == training_size + test_size and cat_count == training_size + test_size:

        break



print("Training and Test Data Extraction Complete")
from keras.models import Sequential

from keras.layers import Activation, Dropout, Flatten, Dense

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

from keras import models

import scipy

import pylab as pl

import matplotlib.cm as cm

import matplotlib.pyplot as plt

%matplotlib inline



input_shape = (150, 150, 3)

img_width = 150

img_height = 150



nb_train_samples = 2000

nb_validation_samples = 1000

batch_size = 32

epochs = 10



train_data_dir = "./datasets/catsvsdogs/train"

validation_data_dir = "./datasets/catsvsdogs/validation"



# Rescale pixel value from (0, 255) to (0, 1)

datagen = ImageDataGenerator(rescale=1./255)



# Automatically retrieve image and classes for training and validation sets

train_generator = datagen.flow_from_directory(train_data_dir,

                                              target_size=(img_width, img_height),

                                              batch_size=16,

                                              class_mode="binary")

validation_generator = datagen.flow_from_directory(validation_data_dir,

                                                   target_size=(img_width, img_height),

                                                   batch_size=32,

                                                   class_mode="binary")



model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))

model.add(Conv2D(32, (3, 3)))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3)))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(64))

model.add(Activation("relu"))

model.add(Dropout(0.5))

model.add(Dense(1))

model.add(Activation("sigmoid"))

print(model.summary())



model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])



train_datagen_augmented = ImageDataGenerator(rescale=1./255,

                                             shear_range=0.2,

                                             zoom_range=0.2,

                                             rotation_range=30,

                                             horizontal_flip=True)

train_datagen_augmented = train_datagen_augmented.flow_from_directory(train_data_dir,

                                                                      target_size=(img_width, img_height),

                                                                      batch_size=batch_size,

                                                                      class_mode="binary")



history = model.fit_generator(train_generator,

                              steps_per_epoch=nb_train_samples // batch_size,

                              epochs=epochs,

                              validation_data=validation_generator,

                              validation_steps=nb_validation_samples // batch_size)
from keras.preprocessing import image



input_image_path = "./datasets/catsvsdogs/validation/cats/cat074.jpg"

img1 = image.load_img(input_image_path)

plt.imshow(img1)



# Load image into 4D tensor and convert it a 4 dim numpy array

img_size = (150, 150)

img1 = image.load_img(input_image_path, target_size=img_size)

image_tensor = image.img_to_array(img1)

image_tensor = image_tensor / 255

image_tensor = np.expand_dims(image_tensor, axis=0)
# Extract top 8 layers

layer_outputs = [layer.output for layer in model.layers[:9]]



# Create model

activation_model = models.Model(input=model.input, outputs=layer_outputs)



# Run prediction function

activations = activation_model.predict(image_tensor)



first_layer_activation = activations[0]

print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 1], cmap="viridis")
plt.matshow(first_layer_activation[0, :, :, 5], cmap="viridis")
for i in range(0, 32):

    plt.matshow(first_layer_activation[0, :, :, i], cmap="viridis")
images_per_row = 16

layer_names = []

for layer in model.layers[:9]:

    layer_names.append(layer.name)



# Retrieve Convolution Layers    

conv_layer_names = []

for layer_name in layer_names:

    if "conv2d" in layer_name:

        conv_layer_names.append(layer_name)

        

for layer_name, layer_activation in zip(conv_layer_names, activations):

    size = layer_activation.shape[1]

    n_features = layer_activation.shape[-1]

    n_cols = n_features // images_per_row

    display_grid = np.zeros((size * n_cols, images_per_row * size))

    

    for col in range(n_cols):

        for row in range(images_per_row):

            channel_image = layer_activation[0, :, :, col * images_per_row + row]

            channel_image -= channel_image.mean()

            channel_image /= channel_image.std()

            channel_image *= 64

            channel_image += 128

            channel_image = np.clip(channel_image, 0, 255).astype("uint8")

            display_grid[col * size:(col + 1) * size, row * size:(row + 1) * size] = channel_image

            

    scale = 1. / size

    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))

    plt.title(layer_name)

    plt.grid(False)

    plt.imshow(display_grid, aspect="auto", cmap="viridis")
shutil.rmtree("./datasets/")