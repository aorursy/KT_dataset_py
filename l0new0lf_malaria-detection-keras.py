# Import libraries

from PIL import Image

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import cv2

import keras

# For one-hot-encoding

from keras.utils import np_utils

# For creating sequenttial model

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

# For saving and loading models

from keras.models import load_model
# data loctation 

os.listdir("/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/")
# Returns list of image file names of class 'Parasitized' and 'Uninfected'

Parasitized = os.listdir("/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized")

Uninfected = os.listdir("/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected")

data = []

labels = []
for filename in Parasitized:

    try:

        image = cv2.imread("/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized/"+filename)

        image_from_numpy_array = Image.fromarray(image, "RGB")

        resized_image = image_from_numpy_array.resize((50, 50))

        data.append(np.array(resized_image))

        labels.append(0)

    except AttributeError:

        print("Attribute error occured for "+filename)

        print("No need to worry for this ttype of error")
for filename in Uninfected:

    try:

        image = cv2.imread("/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected/"+filename)

        image_from_numpy_array = Image.fromarray(image, "RGB")

        resized_image = image_from_numpy_array.resize((50, 50))

        data.append(np.array(resized_image))

        labels.append(1)

    except AttributeError:

        print("Attribute error occured for "+filename)

        print("No need to worry for this ttype of error")
cells = np.array(data)

labels = np.array(labels)
# Saving data as .npy

# Here in labels,

# '0' --> Parasitized

# '1' --> Uniinfected

np.save("all-cells-as-rgb-image-arrays", cells)

np.save("corresponding-labels-for-all-cells", labels)
cells = np.load("all-cells-as-rgb-image-arrays.npy")

labels = np.load("corresponding-labels-for-all-cells.npy")
cells.shape
# Create a numpy array [0, 1, 2 ... 27557]

shuffle = np.arange(cells.shape[0])



# shuffle elements created numpy array

np.random.shuffle(shuffle)



# shuffle 'cells' and 'labels' using 'shuffle' numpy array

cells = cells[shuffle]

labels = labels[shuffle]
num_classes = len(np.unique(labels)) # for test-train-split

len_data = len(cells) #27558 - for keras one hot encoding
# creating test(10%) - train(90%) - split

(x_train,x_test)=cells[(int)(0.1*len_data):],cells[:(int)(0.1*len_data)]

(y_train,y_test)=labels[(int)(0.1*len_data):],labels[:(int)(0.1*len_data)]
# Normalizing data

x_train = x_train.astype("float32") / 255.0

x_test = x_test.astype("float32") / 255.0
# one hot encoding for keras

y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)
# sequential model
model = Sequential()

model.add(Conv2D(filters=16, kernel_size=2, padding="same", activation="relu", input_shape=(50,50,3)))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=2, padding="same", activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(500, activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(2, activation="softmax"))

model.summary()
model.compile(loss="categorical_crossentropy",

               optimizer="adam",

               metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=50, epochs=20, verbose=1)
accuracy =model.evaluate(x_test, y_test, verbose=1)

print(accuracy[1])
# save model weights

model.save("keras-malaria-detection-cnn.h5")
# predict on single image

# use all keras features here