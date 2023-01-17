# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

import cv2

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D

from keras.utils import np_utils

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# path to images

path = '/kaggle/input/animal-image-datasetdog-cat-and-panda/animals/'



# animal categories

categories = ['dogs', 'panda', 'cats']

    
# let's display some of the pictures



for category in categories:

    fig, _ = plt.subplots(3,4)

    fig.suptitle(category)

    for k, v in enumerate(os.listdir(path+category)[:12]):

        img = plt.imread(path+category+'/'+v)

        plt.subplot(3, 4, k+1)

        plt.axis('off')

        plt.imshow(img)

    plt.show()
shape0 = []

shape1 = []



for category in categories:

    for files in os.listdir(path+category):

        shape0.append(plt.imread(path+category+'/'+ files).shape[0])

        shape1.append(plt.imread(path+category+'/'+ files).shape[1])

    print(category, ' => height min : ', min(shape0), 'width min : ', min(shape1))

    print(category, ' => height max : ', max(shape0), 'width max : ', max(shape1))

    shape0 = []

    shape1 = []

# initialize the data and labels

data = []

labels = []

imagePaths = []

HEIGHT = 32

WIDTH = 55

N_CHANNELS = 3



# grab the image paths and randomly shuffle them

for k, category in enumerate(categories):

    for f in os.listdir(path+category):

        imagePaths.append([path+category+'/'+f, k]) # k=0 : 'dogs', k=1 : 'panda', k=2 : 'cats'



import random

random.shuffle(imagePaths)

print(imagePaths[:10])



# loop over the input images

for imagePath in imagePaths:

    # load the image, resize the image to be HEIGHT * WIDTH pixels (ignoring

    # aspect ratio) and store the image in the data list

    image = cv2.imread(imagePath[0])

    image = cv2.resize(image, (WIDTH, HEIGHT))  # .flatten()

    data.append(image)

    

    # extract the class label from the image path and update the

    # labels list

    label = imagePath[1]

    labels.append(label)



# scale the raw pixel intensities to the range [0, 1]

data = np.array(data, dtype="float") / 255.0

labels = np.array(labels)
# Let's check everything is ok

plt.subplots(3,4)

for i in range(12):

    plt.subplot(3,4, i+1)

    plt.imshow(data[i])

    plt.axis('off')

    plt.title(categories[labels[i]])

plt.show()
# partition the data into training and testing splits using 80% of

# the data for training and the remaining 20% for testing

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)
# Preprocess class labels

trainY = np_utils.to_categorical(trainY, 3)



print(trainX.shape)

print(testX.shape)

print(trainY.shape)

print(testY.shape)
model = Sequential()



model.add(Convolution2D(32, (2, 2), activation='relu', input_shape=(HEIGHT, WIDTH, N_CHANNELS)))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, (2, 2), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(3, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



print(model.summary())
model.fit(trainX, trainY, batch_size=32, epochs=25, verbose=1)
from numpy import argmax

from sklearn.metrics import confusion_matrix, accuracy_score



pred = model.predict(testX)

predictions = argmax(pred, axis=1) # return to label



cm = confusion_matrix(testY, predictions)



fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(cm)

plt.title('Model confusion matrix')

fig.colorbar(cax)

ax.set_xticklabels([''] + categories)

ax.set_yticklabels([''] + categories)



for i in range(3):

    for j in range(3):

        ax.text(i, j, cm[j, i], va='center', ha='center')



plt.xlabel('Predicted')

plt.ylabel('True')

plt.show()





accuracy = accuracy_score(testY, predictions)

print("Accuracy : %.2f%%" % (accuracy*100.0))