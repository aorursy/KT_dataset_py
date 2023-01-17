import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import tensorflow as tf

from tensorflow import keras

from tensorflow.python.framework import ops

ops.reset_default_graph()

from tensorflow.keras.utils import multi_gpu_model

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split

import keras.preprocessing.image

import os

import cv2

#Loading data and setting categories 

DATADIR= r"../input/cell_images/cell_images"

CATEGORIES = ['Parasitized','Uninfected']
for i in CATEGORIES:

    path = os.path.join(DATADIR, i) #Path to directory

    for image in os.listdir(path):

        image_array= cv2.imread(os.path.join(path,image), cv2.IMREAD_GRAYSCALE) #Image is in gray scale to lessen load on model

        plt.imshow(image_array, cmap='gray')

        plt.show()

        break

    break
print(image_array.shape)
img_size= 100

new_array= cv2.resize(image_array, (img_size, img_size)) #resizing image 

plt.imshow(new_array)

plt.show()
training_data= []

def create_training_data():

    for i in CATEGORIES:

        path = os.path.join(DATADIR, i) #Path to directory

        class_num= CATEGORIES.index(i)

        for image in os.listdir(path):

            try:

                image_array= cv2.imread(os.path.join(path,image), cv2.IMREAD_GRAYSCALE)

                new_array= cv2.resize(image_array, (img_size, img_size))

                training_data.append([new_array, class_num])

            except Exception as e:

                pass

create_training_data()
import random

random.shuffle(training_data)

for sample in training_data[:10]:

    print(sample[1])
X=[]

y=[]

for features, label in training_data:

    X.append(features)

    y.append(label)

X=np.array(X).reshape(-1, img_size, img_size,1) 
# normalizing data or treating all values as between 0 and 1

X = X/255

print(len(X))

print(len(y))

#making sure they are equal 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5, random_state=42) 
#No need to have the model run for a while so adding a callback to stop training once 97 percent accuracy is reached.

class myCallback (tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('acc')>0.97):

            print('\Reached 97% accuracy so cancelling training!')

            self.model.stop_training= True

            

callbacks= myCallback()
model = Sequential()

model.add(Conv2D(64, (3,3), input_shape= X.shape[1:]))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(64, (3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64))



model.add(Activation('relu'))

model.add(Dense(1))

model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics= ['accuracy'])



model.fit(X, y, batch_size=32, epochs=15, validation_split=0.1, callbacks=[callbacks])
predictions = model.evaluate(X_test, y_test)

print(f'Loss : {predictions[0]}')

print(f'Accuracy : {predictions[1]}')
