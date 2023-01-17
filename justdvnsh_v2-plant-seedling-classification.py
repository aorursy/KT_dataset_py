# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import cv2
# set the matplotlib backend so figures can be saved in the background
import matplotlib .pyplot as plt
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras import backend as K
import random
import os

# Any results you write to the current directory are saved as output.
data = []
labels = []

for path, _, files in os.walk('../input/nonsegmentedv2'):
    imagesPaths =  sorted([image for image in files])
    #print(imagesPaths)
    for imagePath in imagesPaths:
        image = cv2.imread('../input/nonsegmentedv2/'+ path.split('/')[3] + '/' + imagePath)
        image = cv2.resize(image, (32,32)).flatten()
        data.append(image)
        labels.append(path.split('/')[3])
        
data

data = np.array(data, dtype='float32') / 255.0
# since the pixel intensities lies from 0 to 255, thus we normalized the data to 0 to 1
labels = np.array(labels)
data
data.shape
labels.shape
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)
## Here We are just doing the one-hot encoding for the labels.
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
trainY.shape
trainY
model = Sequential()
# First hidden layer , with 1024 nodes and the input shape would be 3072 ,
# since we have made the images into 32x32 shape.
# thus 32x32x3 = 3072 since the images have 3 colour channels , RGB.
# So there are 5539 images each image with 3072 columns or dimentions.
model.add(Dense(1024, input_shape=(3072,), activation = 'sigmoid'))
# the second hidden layer will have 512 nodes
model.add(Dense(512, activation='sigmoid'))
# Now since there are 12 classes, thus , the number of output nodes needs to be the number of classes i.e. 12
model.add(Dense(len(lb.classes_), activation="softmax"))
INIT_LR = 0.01
EPOCHS = 501
# compile the model using SGD as our optimizer and categorical
# cross-entropy loss 
print("[INFO] training network...")
opt = SGD(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
h = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size = 32)
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=lb.classes_))
 
# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.figure()
plt.plot(N, h.history["loss"], label="train_loss")
plt.plot(N, h.history["val_loss"], label="val_loss")
plt.plot(N, h.history["acc"], label="train_acc")
plt.plot(N, h.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
image = cv2.imread('../input/nonsegmentedv2/Maize/1.png')
output = image.copy()
image = cv2.resize(image, (32,32))
image = image.astype('float') / 255
image = image.flatten()
image = image.reshape((1, image.shape[0]))
image.shape
preds = model.predict(image)
i = preds.argmax(axis=1)[0]
label = lb.classes_[i]
print(preds, label)
class SmallVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        
        # If the we are using 'channels first', then we need to update the input shape
        # and channels dimentions
        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)
            chanDim = 1
            
        model.add(Conv2D(32, (3,3), padding = 'same' ,input_shape = inputShape))
        model.add(Activation('relu'))
        ## To normalize the data along the channel dimention, to reduce training time and stabilize
        ## the network.
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # (CONV => RELU) * 2 => POOL layer set
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # (CONV => RELU) * 3 => POOL layer set
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
data = []
labels = []

for path, _, files in os.walk('../input/nonsegmentedv2'):
    imagesPaths =  sorted([image for image in files])
    #print(imagesPaths)
    for imagePath in imagesPaths:
        image = cv2.imread('../input/nonsegmentedv2/'+ path.split('/')[3] + '/' + imagePath)
        image = cv2.resize(image, (64,64))
        data.append(image)
        labels.append(path.split('/')[3])
        
data
data = np.array(data, dtype='float32') / 255.0
# since the pixel intensities lies from 0 to 255, thus we normalized the data to 0 to 1
labels = np.array(labels)
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)
## Here We are just doing the one-hot encoding for the labels.
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
# construct the image generator for data augmentation
# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                        horizontal_flip=True, fill_mode="nearest")
 
"""
Image augmentation allows us to construct “additional” training data from our existing training data by randomly rotating, shifting, shearing, zooming, and flipping.

Data augmentation is often a critical step to:

1) Avoiding overfitting
2) Ensuring your model generalizes well
"""
    
# initialize our VGG-like Convolutional Neural Network
model = SmallVGGNet.build(width=64, height=64, depth=3,classes=len(lb.classes_))
# initialize our initial learning rate, # of epochs to train for,
# and batch size
INIT_LR = 0.01
EPOCHS = 100
BS = 32
# initialize the model and optimizer 
print("[INFO] training network...")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

# train the network
h = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                                validation_data=(testX, testY), 
                                steps_per_epoch=len(trainX) // BS,
                                epochs=EPOCHS)
# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=lb.classes_))
 
# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.figure()
plt.plot(N, h.history["loss"], label="train_loss")
plt.plot(N, h.history["val_loss"], label="val_loss")
plt.plot(N, h.history["acc"], label="train_acc")
plt.plot(N, h.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (SmallVGGNet)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
