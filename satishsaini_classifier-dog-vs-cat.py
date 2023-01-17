Dir='/kaggle/input/cat-and-dog/training_set/training_set'
import os
import glob
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import preprocessing
from keras.utils import to_categorical
img = glob.glob("../input/cat-and-dog/training_set/training_set/*/*.jpg")
print(f"number of images : {len(img)}")
from random import shuffle
shuffle(img)
plt.figure(figsize=(20,20))
for i in range(25):
    plt.subplot(5,5,i+1)
    animal_img = img[i]
    image = plt.imread((animal_img))
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(animal_img[47:50])
    
image_list =[]
label=[]
for i in range(8000):
    animal_img = img[i] # image path
    read_image = preprocessing.image.load_img(animal_img,target_size = (128,128))  # reading image
    image = preprocessing.image.img_to_array(read_image) # converting to array
    
    image_list.append(image) 
    if animal_img[47:50]=="dog":  
        label.append(1)
    else:
        label.append(0)
image_list = np.asarray(image_list)
image_list.shape
label = to_categorical(label,num_classes=2)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Flatten,Dropout,Conv2D,MaxPooling2D
model=Sequential()

model.add(Conv2D(filters=16,kernel_size=3,activation="relu",input_shape=image_list.shape[1:]))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=3,activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=3,activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dense(2,activation="softmax"))


model.summary()
model.compile(loss="categorical_crossentropy",optimizer="Adam",metrics=["accuracy"])
model.fit(x=image_list,y=label,epochs=20)
model.save("model.h5")
Dir = "../input/cat-and-dog/test_set/test_set"
img = glob.glob(Dir+"/*/*.jpg")
print(f"number of images : {len(img)}")
from random import shuffle
shuffle(img)
plt.figure(figsize=(20,20))
for i in range(25):
    plt.subplot(5,5,i+1)
    animal_img = img[i]
    image = plt.imread((animal_img))
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(animal_img[47:50])
