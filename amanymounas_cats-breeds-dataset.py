import os.path
import numpy as np
import cv2
import pickle
import sys
import os
import csv
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from PIL import Image
import pandas as pd

import seaborn as sns
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt



#Useful function to read data files
def createFileList(Mydir, format='.jpg'):
    i=0
    fileList = []
    for breed in os.listdir(Mydir) :
        counts =len(os.listdir(Mydir + breed))
        if counts >= 4000 and counts < 50000:
            i+=1
            for name in os.listdir(Mydir + breed):
                if name.endswith(format):
                    fullName = os.path.join("../input/cat-breeds-dataset/" + "images/" , breed, name)
                    #print(fullName)
                    fileList.append(fullName)
    return fileList, i


# load the original image
breeds = ("../input/cat-breeds-dataset/" + "images/")
myFileList, i = createFileList(breeds)
print(i)


# initialize the data and labels
data   = []
labels = []

# loop over the input images
for image_file in myFileList:

    # Load the image
    image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
   
    # Resize the image so it fits in a 300x300 pixel box
    dim = (300, 300)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    
    # Grab the name of the label based on the folder it was in
    label = image_file.split(os.path.sep)[-2]
    
    # Add the  image and it's label to our training data
    data.append(image)
    labels.append(label)
    
print("finished")
    

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

print(np.shape(data))
print(np.shape(labels))

# Split the training data into separate train and test sets
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)

# Convert the labels (letters) into one-hot encodings that Keras can work with
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)
# Build the neural network!
model = Sequential()

# First convolutional layer with max pooling
model.add(Conv2D(32, (5, 5), padding="same", activation="relu", 
                input_shape=(300,300,3)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
#Dropout to avoid overfitting
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))

#Flatten output of conv
model.add(Flatten())

#Fully Connected layer
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))

# Output layer with 32 nodes (one for each possible letter/number we predict)
model.add(Dense(4, activation="softmax"))

# Ask Keras to build the TensorFlow model behind the scenes
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

# Train the neural network
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=86, epochs=30, verbose=1)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
# compile model
opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=86, epochs=30, verbose=1)
