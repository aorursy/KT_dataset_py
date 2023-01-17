# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#my code starts here

import os
import cv2
import pickle
import matplotlib.pyplot as plt
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Activation,Flatten, Conv2D, MaxPooling2D

#renaming real and fake directories

real = "../input/real-and-fake-face-detection/real_and_fake_face/training_real/"
fake = "../input/real-and-fake-face-detection/real_and_fake_face/training_fake/"


#we're creating a list of real and fake images
real_path = os.listdir(real)
# print (real_path[0:10])
fake_path = os.listdir(fake)
# print (fake_path[0:10])
#let us print real and fake faces

def load_img(path):
    image = cv2.imread(path)
    image = cv2.resize(image,(100,100))
    return image[...,::-1]
#run this code block to see real faces

fig = plt.figure(figsize=(10, 10))

for i in range(25):
    plt.subplot(5,5, i+1)
    plt.imshow(load_img(real + real_path[i]), cmap='gray')
    plt.suptitle("Real faces",fontsize=20)
    plt.axis('off')

plt.show()
#run this code block to see fake faces

fig = plt.figure(figsize=(10, 10))

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(load_img(fake + fake_path[i]), cmap='gray')
    plt.suptitle("Fake faces",fontsize=20)
    plt.axis('off')

plt.show()
#creating training data with label for both genuine and fake images together
#we want to one hot encode our label such that it is in the following format
# [1] = genuine; 
# [0,1] = fake

img_size = int(128)

def create_training_data():
    training_data = []
    for img in tqdm(real_path):
        path = os.path.join(real, img)
        label = [1] 
        image = cv2.resize( cv2.imread(path, cv2.IMREAD_GRAYSCALE), (img_size,img_size) )
        training_data.append([np.array(image), np.array(label)])
        
    for img in tqdm(fake_path):
        path = os.path.join(fake, img)
        label = [0] 
        image = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (img_size,img_size))
        training_data.append([np.array(image), np.array(label)])
        
    
    for img in tqdm(real_path):
        path = os.path.join(real, img)
        label = [1] 
        image = cv2.resize( cv2.imread(path, cv2.IMREAD_GRAYSCALE), (img_size,img_size) )
        training_data.append([np.array(image), np.array(label)])
        
    for img in tqdm(fake_path):
        path = os.path.join(fake, img)
        label = [0] 
        image = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (img_size,img_size))
        training_data.append([np.array(image), np.array(label)])
        
    shuffle(training_data) #shuffle data for more variability in training set
    return(training_data)

#total training data with fake + real images, along with labels, converted to a grayscale
#adding data 2x to populate model more. Doing this bumped up accuracy from 60 to 70%

train_data = create_training_data()

#semantic reshaping of data to feed into model
#separating labels, and features into X and y

X = []
y = []

for i in train_data:
    X.append(i[0])
    y.append(i[1])
        
# print(X[0].reshape(-1, 50, 50, 1))
X = np.array(X).reshape(-1, img_size, img_size, 1)

#divide by 255 to squish values to 0 - 1
X = X/255.0
y = np.array(y)

#we made X and y np arrays to be able to feed into the model
X.shape[1:] #checking shape of array to confirm
# print(len(y))
#let us make the model

model = Sequential()

model.add(Conv2D(128,(3,3), input_shape=X.shape[1:])) 
model.add(Activation("relu")) 
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3))) 
model.add(Activation("relu")) 
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(.5))

model.add(Conv2D(32,(3,3))) 
model.add(Activation("relu")) 
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(2,2))) 
model.add(Activation("relu")) 
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.5))

model.add(Flatten()) 
model.add(Dense(128))

model.add(Dense(1)) 
model.add(Activation("sigmoid"))

opt = keras.optimizers.Adam(learning_rate=0.0002)

model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ['accuracy'])

model.fit(X, y, batch_size = 16, epochs = 70, verbose = 1, validation_split = 0.2)
model.save('my_finalised_cnn', save_format='tf')
model2 = keras.models.load_model('my_finalised_cnn')
print(model2.summary())
#lets test it for real (pun unintended but this is the highlight of my day lol)


def give_img(path):
    img_size = int(128)
    img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (img_size, img_size))
    return img.reshape(-1, img_size, img_size, 1)

pred = model2.predict()
    
