import os

print(os.listdir('../input/gtsrb-german-traffic-sign'))
import os

import cv2

from PIL import Image

import pandas as pd

import numpy as np

import random

import tensorflow as tf

import tensorflow.keras as keras

from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dropout, Dense

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
# Reading the input images and putting them into a numpy array

data=[]

labels=[]



height = 30

width = 30

channels = 3

num_classes = 43

n_inputs = height * width*channels



for i in range(num_classes) :

    path = "../input/gtsrb-german-traffic-sign/train/{0}/".format(i)

    print(path)

    Class=os.listdir(path)

    for a in Class:

        try:

            image=cv2.imread(path+a)

            image_from_array = Image.fromarray(image, 'RGB')

            size_image = image_from_array.resize((height, width))

            data.append(np.array(size_image))

            labels.append(i)

        except AttributeError:

            print(" ")

            

x_train=np.array(data)

x_train= x_train/255.0



y_train=np.array(labels)

y_train = to_categorical(y_train, num_classes) # Using one hote encoding
# Split Data

X_train,X_valid,Y_train,Y_valid = train_test_split(x_train,y_train,test_size = 0.3,random_state=0)

print("Train :", X_train.shape)

print("Valid :", X_valid.shape)
# Show Train images

import matplotlib.pyplot as plt



def show_images(images, labels, amount):

    for i in range(amount):

        index = int(random.random() * len(images))

        plt.axis('off')

        plt.imshow(images[index])

        plt.show()       

        print("Size of this image is " + str(images[index].shape))

        print("Class of the image is " + str(labels[index]))



print("Train images")

show_images(X_train, Y_train, 3)
# Build Model

model = keras.models.Sequential()



model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='relu', input_shape=X_train.shape[1:]))

model.add(Conv2D(32, kernel_size=(3,3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))



model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))

model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))



model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(rate=0.5))

model.add(Dense(num_classes, activation='softmax'))



model.summary()
# Compile Model

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train Model

epochs = 10

history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), batch_size=32, epochs=epochs,verbose=1)
# plot the accuracy and the loss

import matplotlib.pyplot as plt



plt.figure(0)

plt.plot(history.history['accuracy'], label='training accuracy')

plt.plot(history.history['val_accuracy'], label='val accuracy')

plt.title('Accuracy')

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.legend()



plt.figure(1)

plt.plot(history.history['loss'], label='training loss')

plt.plot(history.history['val_loss'], label='val loss')

plt.title('Loss')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.legend()
# Predicting Test data 

y_test=pd.read_csv("../input/gtsrb-german-traffic-sign/Test.csv")

labels=y_test['Path'].as_matrix()

y_test=y_test['ClassId'].values



data=[]



for f in labels:

    image=cv2.imread('../input/gtsrb-german-traffic-sign/test/'+f.replace('Test/', ''))

    image_from_array = Image.fromarray(image, 'RGB')

    size_image = image_from_array.resize((height, width))

    data.append(np.array(size_image))



X_test=np.array(data)

X_test = X_test.astype('float32')/255  

pred = model.predict_classes(X_test)
# Accuracy with the test data

from sklearn.metrics import accuracy_score

accuracy_score(y_test, pred)