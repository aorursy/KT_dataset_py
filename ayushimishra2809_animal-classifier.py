import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten ,Dropout, Conv2D, MaxPooling2D, BatchNormalization

from tensorflow.keras.models import Sequential

import numpy as np

import os 

import random

import cv2

from sklearn.model_selection import train_test_split
categories = {'cane': 'dog', "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel","ragno":"spider"}

data=[]

animals=["dog", "horse","elephant", "butterfly",  "chicken",  "cat", "cow",  "sheep", "squirrel","spider"]

img_size=100

def create_data():

        for category,translate in categories.items():

            path="../input/animals10/raw-img/"+category

            target=animals.index(translate)

            

            for img in os.listdir(path):

                try:

                    img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)

                    new_img_array=cv2.resize(img_array,(img_size,img_size))

                    data.append([new_img_array,target])

                except Exception as e:

                    pass

                

            

create_data()            
random.shuffle(data)

x=[]

y=[]

for features,labels in data:

    x.append(features)

    y.append(labels)   

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


x_train=np.array(x_train).reshape(-1,img_size,img_size,1)

x_train=tf.keras.utils.normalize(x_train,axis=1)

y_train=np.array(y_train)

# model=Sequential()

# model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = x_train.shape[1:]))

# model.add(BatchNormalization())

# model.add(Conv2D(32, kernel_size = 3, activation='relu'))

# model.add(BatchNormalization())

# model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))

# model.add(BatchNormalization())

# model.add(Dropout(0.4))

# model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))

# model.add(BatchNormalization())

# model.add(Dropout(0.4))

# model.add(Conv2D(256, kernel_size = 4, activation='relu'))

# model.add(BatchNormalization())

# model.add(Flatten())

# model.add(Dropout(0.4))

# model.add(Dense(64, activation='softmax'))          

# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])          


# model.fit(x_train,y_train,epochs=50,batch_size=1000)

# prediction=model.predict(np.array(x_test).reshape(-1,img_size,img_size,1))

# model.score(y_test,prediction)