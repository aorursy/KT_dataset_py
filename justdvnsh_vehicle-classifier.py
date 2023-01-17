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
from sklearn import svm
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
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
from keras.utils import to_categorical
from keras.models import model_from_json
import random

# Any results you write to the current directory are saved as output.
data = []
labels = []

for paths, dirs,files in os.walk('../input/owncollection'):
    imagesPath = sorted([images for images in files])
    #print(imagesPath)
    #print(paths)
    for imagePath in imagesPath:
        image = cv2.imread('../input/owncollection/OwnCollection/' + paths.split('/')[4] + '/' + paths.split('/')[5] + '/' + imagePath)
        image = cv2.resize(image, (64,64)).flatten()
        data.append(image)
        labels.append(paths.split('/')[4])

print(len(data))
print(len(labels))
data = np.array(data, dtype='float32') / 255.0
labels = np.array(labels)
data.shape
labels.shape
data[:100]
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)
trainY.shape
testY.shape
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
trainY
print(trainY.shape)
print(testY.shape)
clf = svm.SVC(kernel='rbf', gamma = 'auto',random_state=0)
clf.fit(trainX, trainY)
image = cv2.imread('../input/owncollection/OwnCollection/vehicles/Right/image0001.png')
image = cv2.resize(image, (64,64)).flatten()
image = image.astype('float') / 255
image = image.reshape((1, image.shape[0]))
image.shape
preds = clf.predict(image)
print(preds)
label = lb.classes_[preds]
print(label)
y_pred = clf.predict(testX)
y_pred.shape
y_pred[:10]
cm = confusion_matrix(testY, y_pred)
print(cm)
score = accuracy_score(testY, y_pred)
score
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

for paths, dirs,files in os.walk('../input/owncollection'):
    imagesPath = sorted([images for images in files])
    #print(imagesPath)
    #print(paths)
    for imagePath in imagesPath:
        image = cv2.imread('../input/owncollection/OwnCollection/' + paths.split('/')[4] + '/' + paths.split('/')[5] + '/' + imagePath)
        image = cv2.resize(image, (64,64))
        data.append(image)
        labels.append(paths.split('/')[4])

print(len(data))
print(len(labels))


data[:3]
data = np.array(data, dtype='float32') / 255.0
# since the pixel intensities lies from 0 to 255, thus we normalized the data to 0 to 1
labels = np.array(labels)
print(data.shape)
print(labels.shape)
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)
## Here We are just doing the one-hot encoding for the labels.
labelencoder_y_1 = LabelEncoder()
trainY = labelencoder_y_1.fit_transform(trainY)
testY = labelencoder_y_1.transform(testY)
print(trainY.shape, testY.shape)
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
model = SmallVGGNet.build(width=64, height=64, depth=3,classes=len(labelencoder_y_1.classes_))
# initialize our initial learning rate, # of epochs to train for,
# and batch size
INIT_LR = 0.01
EPOCHS = 100
BS = 32
# initialize the model and optimizer 
print("[INFO] training network...")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

# train the network
h = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                                validation_data=(testX, testY), 
                                steps_per_epoch=len(trainX) // BS,
                                epochs=EPOCHS)
# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY,
                            predictions.argmax(axis=1), target_names=labelencoder_y_1.classes_))
 
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
image = cv2.imread('../input/owncollection/OwnCollection/non-vehicles/Far/image0000.png')
image = cv2.resize(image, (64,64))
image = image.astype('float') / 255.0
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image.shape
preds = model.predict(image)
i = preds.argmax(axis=1)[0]
label = labelencoder_y_1.classes_[i]
print(preds, label)
image = cv2.imread('../input/owncollection/OwnCollection/vehicles/Far/image0000.png')
image = cv2.resize(image, (64,64))
image = image.astype('float') / 255.0
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image.shape
preds = model.predict(image)
i = preds.argmax(axis=1)[0]
label = labelencoder_y_1.classes_[i]
print(preds, label)
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
