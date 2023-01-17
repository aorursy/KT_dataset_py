# Ignore  the warnings

import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')



# specifically for manipulating images and getting numpy arrays of pixel values of images.

import cv2                  

import numpy as np  

from tqdm import tqdm

import os       

import random

from random import shuffle 

import tqdm





# data visualisation

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns





#model selection

from sklearn.preprocessing import LabelEncoder



# specifically for cnn

from keras.layers import Dropout, Flatten,Activation

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

 
DIR = "../input/flowers/flowers/"

CATEGORIES = os.listdir("../input/flowers/flowers/")

CATEGORIES
images = []

categories = []

IMG_SIZE = 100



for catg in tqdm.tqdm(CATEGORIES):

    path = os.path.join(DIR,catg)

    for img_path in os.listdir(path):

        img_path = os.path.join(path, img_path)

        try:

            img = cv2.imread(img_path)

            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

            img = img / 255.0

            

            images.append(img)

            categories.append(catg)       

        except:

            continue
print(len(images), len(categories))
fig,ax=plt.subplots(5,2)

fig.set_size_inches(15,15)

for i in range(5):

    for j in range (2):

        l=random.randint(0,len(categories))

        ax[i,j].imshow(images[l])

        ax[i,j].set_title('Flower: '+categories[l])

        

plt.tight_layout()
X = np.array(images)

le=LabelEncoder()

y=le.fit_transform(categories)

y=to_categorical(y,5)

print(X.shape, y.shape)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle= True)
from keras.models import Sequential

from keras.layers import Conv2D,MaxPool2D,Flatten,Dense



model = Sequential()

model.add(Conv2D(16, (3,3),input_shape=(IMG_SIZE, IMG_SIZE, 3),activation='relu', padding = 'Same'))

model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(32, (3,3),activation='relu', padding = 'Same'))

model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(64, (3,3),activation='relu', padding = 'Same'))

model.add(MaxPool2D(pool_size=(2,2)))



model.add(Flatten())

model.add(Dropout(0.2))



model.add(Dense(128,activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(5,activation='softmax'))



model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



model.summary()
y_train.shape
batch_size = 128

epochs = 30



model.fit(x_train, y_train,

          batch_size=batch_size,

          verbose = 1,

          epochs=epochs,

          validation_data=(x_test, y_test))