# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import keras

from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

from keras.models import Sequential

import cv2

import sklearn

from sklearn.model_selection import train_test_split
imagens =[]

labels = []

shape=(556,556)



#add daisy

for i in os.listdir('../input/flowers/flowers/daisy/'):

    img=cv2.imread(os.path.join('../input/flowers/flowers/daisy/',i))

    img2=cv2.resize(img,shape)

    imagens.append(img2)

    labels.append('daisy')
#add dandelion

for i in os.listdir('../input/flowers/flowers/dandelion/'):

    img=cv2.imread(os.path.join('../input/flowers/flowers/dandelion/',i))

    if i.split('.')[1] == 'jpg':

        img2=cv2.resize(img,shape)

        imagens.append(img2)

        labels.append('dandelion')
for i in os.listdir('../input/flowers/flowers/rose/'):

    img=cv2.imread(os.path.join('../input/flowers/flowers/rose/',i))

    img2=cv2.resize(img,shape)

    imagens.append(img2)

    labels.append('rose')
for i in os.listdir('../input/flowers/flowers/sunflower/'):

    img=cv2.imread(os.path.join('../input/flowers/flowers/sunflower/',i))

    img2=cv2.resize(img,shape)

    imagens.append(img2)

    labels.append('sunflower')
for i in os.listdir('../input/flowers/flowers/tulip/'):

    img=cv2.imread(os.path.join('../input/flowers/flowers/tulip/',i))

    img2=cv2.resize(img,shape)

    imagens.append(img2)

    labels.append('tulip')
plt.imshow(imagens[500])

plt.title(labels[500])

def model_load(input_shape, output):

    model=Sequential()

    model.add(Conv2D(kernel_size=(3,3), activation='tanh', filters=32, input_shape=input_shape))

    model.add(MaxPool2D(pool_size=(2,2)))

    

    model.add(Conv2D(kernel_size=(3,3), activation='tanh', filters=64))

    model.add(MaxPool2D(pool_size=(2,2)))

    

    model.add(Flatten())

    

    model.add(Dense(units=32, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(units=output, activation='softmax'))

    

    model.compile(loss='categorical_crossentropy',optimizer='sgd')

    

    return model
imagens = np.array(imagens)

input_shape = imagens[0].shape
labels = pd.get_dummies(labels).values

output=len(labels[0])
model = model_load(input_shape, output)
x_train, x_test, y_train, y_test = train_test_split(imagens, labels)
model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test))