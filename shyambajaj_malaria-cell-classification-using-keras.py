#Importing Libraries

import numpy as np

import pandas as pd

import cv2

from PIL import Image

import os
W = 32

H = 32



dataset = []

label = []

infected = os.listdir("../input/cell_images/cell_images/Parasitized")

uninfected = os.listdir("../input/cell_images/cell_images/Uninfected")

for i in infected:

    x = len(i)

    if i[x-3]=='p' and i[x-2]=='n' and i[x-1]=='g':

        try:

            image = cv2.imread("../input/cell_images/cell_images/Parasitized/"+i)

            image = cv2.resize(image,(W,H))

            dataset.append(image/255)

            label.append(1)

            #cv2.imshow('original',image)

            img = Image.fromarray(image)

            

            clockwise = np.array(img.rotate(-45))

            dataset.append(clockwise/255)

            label.append(1)

            #cv2.imshow('clockwise',clockwise)

            

            anticlockwise = np.array(img.rotate(+45))

            dataset.append(anticlockwise/255)

            label.append(1)

            #cv2.imshow('anticlockwise',anticlockwise)

        except AttributeError:

            print("error")

for i in uninfected:

    x = len(i)

    if i[x-3]=='p' and i[x-2]=='n' and i[x-1]=='g':

        try:

            image = cv2.imread("../input/cell_images/cell_images/Uninfected/"+i)

            image = cv2.resize(image,(W,H))

            dataset.append(image/255)

            label.append(0)

            #cv2.imshow('original',image)

            img = Image.fromarray(image)

            

            clockwise = np.array(img.rotate(-45))

            dataset.append(clockwise/255)

            label.append(0)

            #cv2.imshow('clockwise',clockwise)

            

            anticlockwise = np.array(img.rotate(+45))

            dataset.append(anticlockwise/255)

            label.append(0)

            #cv2.imshow('anticlockwise',anticlockwise)

        except AttributeError:

            print("error")


#Converting List to numpy array

X = np.asarray(dataset)

y = np.asarray(label)
#Splitting dataset into Train and Test sets

from sklearn.model_selection import train_test_split

XTrain, XTest, yTrain, yTest = train_test_split(X,y,test_size = 0.4)

print("Training samples = ",len(XTrain))

print("Testing samples = ",len(XTest))
#Importing ANN libraries

from keras.models import Sequential

from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense

from keras.layers.normalization import BatchNormalization
#Inializing CNN

classifier = Sequential()



#Adding 1st Convolution Layer

classifier.add(Convolution2D(filters=32, kernel_size=(3,3), strides=(1,1),input_shape=(H,W,3), activation='relu'))

#Adding 1st MaxPooling Layer to reduce the size of Features

classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



#Adding Batch Normalization for Higher Learnig Rate

classifier.add(BatchNormalization())

#Adding Dropout Layer to eliminate Overfitting

classifier.add(Dropout(0.2))



#Adding 2nd Convolution Layer

classifier.add(Convolution2D(filters=32, kernel_size=(3,3), activation='relu'))

#Adding 2nd MaxPooling Layer

classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))



#Adding Flatten Layer to convert 2D matrix into an array

classifier.add(Flatten())



#Adding 1st Fully Connected Layer

classifier.add(Dense(units=64, activation='relu'))

#Adding Fully Connected Output Layer

classifier.add(Dense(units=1, activation='sigmoid'))

#Compiling the CNN

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



#Fitting the model

history = classifier.fit(XTrain, yTrain, batch_size=10, epochs=10)

from matplotlib import pyplot as plt

plt.plot(history.history['acc'],'green')

plt.plot(history.history['loss'],'red')

plt.title('Model Accuracy-Loss')

plt.xlabel('Epoch')

plt.legend(['Accuracy','Loss'])

plt.figure()

from sklearn.metrics import confusion_matrix, accuracy_score

#Predicting the Test set Results

yPred = classifier.predict(XTest)

yPred = (yPred>0.5) #Since output is probability

cm = confusion_matrix(yTest,yPred)

accuracy = accuracy_score(yTest,yPred)

print("Artificial Neural Network Classifier :")

print("Accuracy = ", accuracy)

print(cm)


