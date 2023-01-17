import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import tensorflow



from tensorflow.keras import datasets
#import data from keras

data =datasets.cifar10.load_data()
#convert into train and test

(train_images, train_labels), (test_images, test_labels) = data
print(train_images.shape)

test_images.shape
train_images,test_images=train_images/255.0 ,test_images/255.0
train_images.shape
train_labels
class_names =['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
#first 25 train_images

plt.figure(figsize =(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.imshow(train_images[i])

    plt.xlabel(class_names[train_labels[i][0]])

    
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D,MaxPooling2D,Activation,Dense,Flatten
model =Sequential()

model.add(Conv2D(32,(3,3),activation ='relu',input_shape =(32,32,3)))

model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3),activation ='relu'))

model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3),activation ='relu'))

model.summary()
model.add(Flatten(input_shape=(32,32,3)))

model.add(Dense(64,activation ='relu'))

model.add(Dense(32,activation ='relu'))

model.add(Dense(10,activation ='softmax'))
model.summary()
model.compile(optimizer='adam',loss ='sparse_categorical_crossentropy' ,metrics =['accuracy'])

model.fit(train_images ,train_labels ,epochs=10 ,validation_data=(test_images,test_labels))
plt.plot(model.history.history['accuracy'],label="accuracy")

plt.plot(model.history.history['val_accuracy'],label ="val_accuracy")

plt.legend()
#our prediction

pred =model.predict(test_images)
class_names[np.argmax(pred[4459])]
#check on some test images

plt.figure(figsize=(10,10))

for i in  range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.imshow(test_images[50+i])

    plt.xlabel(class_names[np.argmax(pred[i+50])])

    

