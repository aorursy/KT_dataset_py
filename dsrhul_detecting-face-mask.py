#import libraries

import numpy as np

import pandas as pd 

import cv2

import matplotlib.pyplot as plt

import os
#sample images

filenames = (os.listdir("../input/face-mask-detection-data/with_mask"))

for i in filenames:

    img = plt.imread("../input/face-mask-detection-data/with_mask/"+i)

    plt.imshow(img)

    plt.title("With mask")

    break
#sample images

filenames = (os.listdir("../input/face-mask-detection-data/without_mask"))

for i in filenames:

    img = plt.imread("../input/face-mask-detection-data/without_mask/"+i)

    plt.imshow(img)

    plt.title("Without mask")

    break
#read images from the dataset

from keras.preprocessing.image import load_img, img_to_array

from numpy import asarray

from PIL import Image

x = []

y = []

filenames = (os.listdir("../input/face-mask-detection-data/with_mask"))

for i in filenames:

    photo = load_img("../input/face-mask-detection-data/with_mask/"+i, target_size=(128,128))

    photo = img_to_array(photo)

    x.append(photo)

    y.append(1) # 1 with mask

filenames = (os.listdir("../input/face-mask-detection-data/without_mask"))

for i in filenames:

    photo = load_img("../input/face-mask-detection-data/without_mask/"+i, target_size=(128,128))

    photo = img_to_array(photo)

    x.append(photo)

    y.append(0) # 0 without mask

x = asarray(x)
#split the data into train and test

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
#import CNN libraries

from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers.normalization import BatchNormalization

from keras.layers.advanced_activations import LeakyReLU

from keras.layers import Conv2D
print(x_train.shape, x_test.shape) 
#convert dependent variable to categorical

from keras.utils import to_categorical

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
#normalize data

x_train = x_train/255.0

x_test =  x_test/255.0
#build CNN model

num_classes = 2

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(128,128,3),padding='same'))

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D((2, 2),padding='same'))

model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))

model.add(LeakyReLU(alpha=0.1))                  

model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

model.add(Flatten())

model.add(Dense(128, activation='linear'))

model.add(LeakyReLU(alpha=0.1))                  

model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
#train model with training set

model_train = model.fit(x_train, y_train, batch_size=64,epochs=20,verbose=1,validation_split=0.20)
#model performance evaluation

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

preds = model.predict(x_test, verbose=1)

print("Accuracy score: ",accuracy_score(y_test, np.round_(preds)))

print("Classification report:")

print(classification_report(y_test, np.round_(preds)))
#results

%matplotlib inline

import random

res = ["Without Mask","With Mask"]

for i in range(5):

    plt.figure()

    im = random.randint(1,958)

    plt.imshow(x_test[im])

    plt.title("Predicted: {},   Actual: {}".format(res[np.argmax(preds[im])], res[np.argmax(y_test[im])]))
#for future use save the model

import pickle

filename = 'finalized_model.sav'

pickle.dump(model, open(filename, 'wb'))