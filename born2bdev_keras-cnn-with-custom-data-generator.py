from keras.preprocessing.image import load_img, ImageDataGenerator

from keras.layers import Conv2D, Dense, MaxPooling2D, Activation, Dropout, Flatten

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from glob import glob

from random import shuffle



import numpy as np

import pandas as pd

import os

import cv2

import matplotlib.pyplot as plt



print(os.listdir("../input"))

print(os.listdir("../input/chest_xray/chest_xray/"))
path_to_data = "../input/chest_xray/chest_xray/"
img_normal_path = path_to_data + "train/NORMAL/NORMAL2-IM-0927-0001.jpeg"

img_normal = load_img(img_normal_path)

print("Normal Image")

plt.imshow(img_normal)

plt.show()
img_other_path = path_to_data + "train/PNEUMONIA/person478_virus_975.jpeg"

img_other = load_img(img_other_path)

print("PNEUMONIA Image")

plt.imshow(img_other)

plt.show()
img_width, img_height = 128, 128

batch_size = 16

epochs = 10
source_images = []

for key in os.listdir(os.path.join(path_to_data, "train", "NORMAL")):

    if not "DS_Store" in key:

        source_images.append(os.path.join(path_to_data, "train", "NORMAL", key))

for key in os.listdir(os.path.join(path_to_data, "train", "PNEUMONIA")):

    if not "DS_Store" in key:

        source_images.append(os.path.join(path_to_data, "train", "PNEUMONIA", key))

shuffle(source_images)

print(len(source_images))
valid_images = []

for key in os.listdir(os.path.join(path_to_data, "val", "NORMAL")):

    if not "DS_Store" in key:

        valid_images.append(os.path.join(path_to_data, "val", "NORMAL", key))

for key in os.listdir(os.path.join(path_to_data, "val", "PNEUMONIA")):

    if not "DS_Store" in key:

        valid_images.append(os.path.join(path_to_data, "val", "PNEUMONIA", key))

shuffle(valid_images)

print(len(valid_images))
test_images = []

for key in os.listdir(os.path.join(path_to_data, "test", "NORMAL")):

    test_images.append(os.path.join(path_to_data, "test", "NORMAL", key))

for key in os.listdir(os.path.join(path_to_data, "test", "PNEUMONIA")):

    test_images.append(os.path.join(path_to_data, "test", "PNEUMONIA", key))

shuffle(test_images)

print(len(test_images))
# Getting number of training, validation and test samples

nb_train_samples = len(source_images)

nb_test_samples = len(test_images)

nb_valid_samples = len(valid_images)



print("Training samples: " + str(nb_train_samples))

print("Validation samples: " + str(nb_valid_samples))

print("Testing samples: " + str(nb_test_samples))
model = Sequential()



model.add(Conv2D(32, (3, 3), input_shape= (img_width, img_height, 3), name="conv1"))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(32, (3, 3), name="conv2"))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(64, (3, 3), name="conv3"))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(64, (3, 3), name="conv4"))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Flatten())



model.add(Dense(128))

model.add(Dropout(0.5))

model.add(Activation("relu"))



model.add(Dense(64))

model.add(Dropout(0.5))

model.add(Activation("relu"))



model.add(Dense(1))

model.add(Activation("sigmoid"))



model.summary()
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
def train_generator():

    while True:

        for start in range(0, nb_train_samples, batch_size):

            x_batch = []

            y_batch = []

            

            end = min(start + batch_size, nb_train_samples)

            for img_path in range(start, end):

                img = cv2.imread(source_images[img_path])

                img = cv2.resize(img, (img_width, img_height))

                x_batch.append(img)

                if "NORMAL" in source_images[img_path]:

                    y_batch.append(["0"])

                elif "PNEUMONIA" in source_images[img_path]:

                    y_batch.append(["1"])

            

            yield (np.array(x_batch), np.array(y_batch))

            

            
def valid_generator():

    while True:

        for start in range(0, nb_valid_samples, batch_size):

            x_batch = []

            y_batch = []

            

            end = min(start + batch_size, nb_valid_samples)

            for img_path in range(start, end):

                img = cv2.imread(valid_images[img_path])

                img = cv2.resize(img, (img_width, img_height))

                x_batch.append(img)

                if "NORMAL" in valid_images[img_path]:

                    y_batch.append(["0"])

                elif "PNEUMONIA" in valid_images[img_path]:

                    y_batch.append(["1"])

            yield (np.array(x_batch), np.array(y_batch))

            

            
def test_generator():

    while True:

        for start in range(0, nb_test_samples, batch_size):

            x_batch = []

            y_batch = []

            

            end = min(start + batch_size, nb_test_samples)

            for img_path in range(start, end):

                img = cv2.imread(test_images[img_path])

                img = cv2.resize(img, (img_width, img_height))

                x_batch.append(img)

                if "NORMAL" in test_images[img_path]:

                    y_batch.append(["0"])

                elif "PNEUMONIA" in test_images[img_path]:

                    y_batch.append(["1"])

            yield (np.array(x_batch), np.array(y_batch))
model.fit_generator(

    train_generator(),

    epochs= epochs,

    steps_per_epoch= nb_train_samples // batch_size,

    validation_data= test_generator(),

    validation_steps = nb_valid_samples // batch_size,

)
model.save_weights("model_1.h5")
# Evaluating the model

scores = model.evaluate_generator(generator=test_generator(), steps=nb_test_samples // batch_size)

print("Test accuracy is {}".format(scores[1] * 100))