# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import keras

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

from PIL import Image

import cv2
img=[]

labels=[]
Parasitized=os.listdir("../input/cell_images/cell_images/Parasitized/")

for a in Parasitized:

    try:

        image=cv2.imread("../input/cell_images/cell_images/Parasitized/"+a)

        image_from_array = Image.fromarray(image, 'RGB')

        size_image = image_from_array.resize((50, 50))

        img.append(np.array(size_image))

        labels.append(0)

    except AttributeError:

        print("")



Uninfected=os.listdir("../input/cell_images/cell_images/Uninfected/")

for b in Uninfected:

    try:

        image=cv2.imread("../input/cell_images/cell_images/Uninfected/"+b)

        image_from_array = Image.fromarray(image, 'RGB')

        size_image = image_from_array.resize((50, 50))

        img.append(np.array(size_image))

        labels.append(1)

    except AttributeError:

        print("")
len(img)
len(labels)
img=np.array(img)

labels=np.array(labels)
img.shape
len_data=len(img)
s=np.arange(img.shape[0])

np.random.shuffle(s)

img=img[s]

labels=labels[s]
(x_train,x_test)=img[(int)(0.1*len_data):],img[:(int)(0.1*len_data)]

x_train = x_train.astype('float32')/255 

x_test = x_test.astype('float32')/255

(y_train,y_test)=labels[(int)(0.1*len_data):],labels[:(int)(0.1*len_data)]
y_train=np_utils.to_categorical(y_train)
y_train
y_test=np_utils.to_categorical(y_test)
model=Sequential()

model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(500,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(250,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(2,activation="softmax"))#2 represent output layer neurons 

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=50,epochs=30,verbose=1,validation_split=0.1)
pred=model.predict(x_test)
pred
pred=np.argmax(pred,axis=1)
pred=np_utils.to_categorical(pred)
pred
y_test
from sklearn.metrics import accuracy_score

accuracy_score(y_test, pred)