from PIL import Image

import numpy as np

import os

import cv2

import keras

from keras.utils import np_utils

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator 

import tensorflow.keras.layers as Layers

import tensorflow.keras.models as Models

import tensorflow.keras.optimizers as Optimizers

import sklearn.utils as shuffle

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
Images = []

Labels = []

Parasitized = os.listdir("../input/cell-images-for-detecting-malaria/cell_images/Parasitized/")

for p in Parasitized:

    try:

        image = cv2.imread("../input/cell-images-for-detecting-malaria/cell_images/Parasitized/" + p)

        images = Image.fromarray(image, 'RGB')

        images = images.resize((150,150))

        Images.append(np.array(images))

        Labels.append(0)

    except AttributeError:

        print('')

Uninfected = os.listdir("../input/cell-images-for-detecting-malaria/cell_images/Uninfected/")

for u in Uninfected:

    try:

        image = cv2.imread("../input/cell-images-for-detecting-malaria/cell_images/Uninfected/" + u)

        images = Image.fromarray(image, 'RGB')

        images = images.resize((150,150))

        Images.append(np.array(images))

        Labels.append(1)

    except AttributeError:

        print('')
Images = np.array(Images)

Labels = np.array(Labels)    
print(Images.shape)

print(Labels.shape)
def show_images(image, label):

    fig = plt.figure(figsize = (10,10))

    fig.suptitle('25 Images from the dataset' ,fontsize = 20)

    for i in range(25):

        index = np.random.randint(Images.shape[0])

        plt.subplot(5,5,i+1)

        plt.imshow(image[index])

        plt.xticks([]) #Scale doesn't appear

        plt.yticks([]) #Scale doesn't apper

        plt.title((label[index]))

        plt.grid(False)

    plt.show()
show_images(Images, Labels)
category = ['Uninfected', 'Parasitized']

_,count = np.unique(Labels, return_counts = True)

pd.DataFrame({'data': count}, index = category).plot.bar()

plt.show()
Labels = keras.utils.to_categorical(Labels, 2)
train_x,test_x, train_y, test_y = train_test_split(Images,Labels, test_size = 0.4, random_state = 100)
model = Models.Sequential()

model.add(Layers.Conv2D(64, kernel_size = (3,3), activation = 'relu', input_shape = (150,150,3)))

model.add(Layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'))

model.add(Layers.MaxPool2D(3,3))

model.add(Layers.Dropout(0.2))

model.add(Layers.Conv2D(64, kernel_size = (3,3) , activation = 'relu'))

model.add(Layers.Conv2D(64, kernel_size = (3,3) , activation = 'relu'))

model.add(Layers.MaxPool2D(3,3))

model.add(Layers.Conv2D(64, kernel_size = (3,3) , activation = 'relu'))

model.add(Layers.Conv2D(64, kernel_size = (3,3) , activation = 'relu'))

model.add(Layers.MaxPool2D(3,3))

model.add(Layers.Flatten())

model.add(Layers.Dense(512, activation = 'relu'))

model.add(Layers.Dense(256, activation = 'relu'))

model.add(Layers.Dropout(0.2))

model.add(Layers.Dense(2, activation = 'softmax'))

model.compile(optimizer = Optimizers.RMSprop(lr=0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.summary()



trained = model.fit(train_x, train_y, epochs = 20, batch_size = 50, validation_split = 0.20, verbose = 1)
plt.plot(trained.history['accuracy'])

plt.plot(trained.history['val_accuracy'])

plt.title("Model Accuracy")

plt.ylabel("Accuracy")

plt.xlabel("Epoch")

plt.legend(["Train", "Validation"], loc = "upper left")

plt.show()
plt.plot(range(20), trained.history['accuracy'], label = 'Training Accuracy')

plt.plot(range(20), trained.history['loss'], label = 'Taining Loss')

plt.xlabel("Number of Epoch's")

plt.ylabel('Accuracy/Loss Value')

plt.title('Training Accuracy and Training Loss')

plt.legend(loc = "best")

plt.show()
result = model.evaluate(test_x,test_y,verbose = 1)
print("Test Accuracy: " , result[1] * 100)