# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2

from tqdm import tqdm

from PIL import Image

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.models import Model

from keras.layers import Dense

from keras.layers import Input

from keras.layers import Conv2D

from keras.layers import Flatten

from keras.layers import Dropout

from keras.utils import plot_model

from keras.models import Sequential

from keras.layers import MaxPooling2D

from keras.utils import to_categorical

from keras.layers import ZeroPadding2D

from keras.layers.merge import concatenate

from keras.preprocessing.image import ImageDataGenerator
import gc
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
train['diagnosis'].hist()

train['diagnosis'].value_counts()
def display_samples(df, columns=4, rows=3):

    fig=plt.figure(figsize=(5*columns, 4*rows))



    for i in range(columns*rows):

        image_path = df.loc[i,'id_code']

        image_id = df.loc[i,'diagnosis']

        img = cv2.imread(f'../input/train_images/{image_path}.png')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        img = cv2.resize(img,(28,28),interpolation=cv2.INTER_CUBIC)

        

        fig.add_subplot(rows, columns, i+1)

        plt.title(image_id)

        plt.imshow(img)

    

    plt.tight_layout()



display_samples(train)
images = []

paths = train.id_code

for path in paths :

    img = cv2.imread(f'../input/train_images/{path}.png')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img,(112,112),interpolation=cv2.INTER_CUBIC).tolist()

    images.append(img)
gc.collect()
model = Sequential()

model.add(ZeroPadding2D((1,1),input_shape=(112,112,1)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))

model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))

model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))

model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(512,kernel_size=(3, 3), activation='relu'))

model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))

model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))

model.add(ZeroPadding2D((1,1)))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())

model.add(Dense(4096, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(4096, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(5, activation='softmax'))
images = np.array(images)
images.shape
trainx, testx, trainy, testy = train_test_split(images,train.diagnosis,test_size=0.2)
trainy= to_categorical(trainy)

testy = to_categorical(testy)
trainx= trainx.reshape(-1, 112, 112,1)

testx= testx.reshape(-1, 112, 112,1)
del images

gc.collect()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainx, trainy,

           epochs=10, batch_size=100, verbose=2)
model.evaluate(testx, testy,

            batch_size=100)