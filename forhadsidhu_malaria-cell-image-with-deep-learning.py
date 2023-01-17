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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import cv2

from PIL import Image





def load_images_from_folder(folder,id,l,w):

    

    for filename in os.listdir(folder):

        if filename!="Thumbs.db":

            img=cv2.imread(os.path.join(folder,filename))

            #print(img.shape)

            if img.shape[0]<l:

                    l=img.shape[0]

            if img.shape[1]<w:

                    w=img.shape[1]

    return l,w

                    

 



l,w=load_images_from_folder("../input/cell_images/cell_images/Parasitized/",2,1000000,1000000)

l,w=load_images_from_folder("../input/cell_images/cell_images/Uninfected/",2,l,w)



print(l,w)
!pip install python-resize-image
from resizeimage import resizeimage

images=[]

labels=[]

def load_images_from_folder_a(folder,id):

    

    for filename in os.listdir(folder):

        if filename!="Thumbs.db":

            img1 = Image.open(os.path.join(folder,filename))

            new1 = resizeimage.resize_contain(img1, [40, 46, 3])

            new1 = np.array(new1, dtype='uint8')

            images.append(new1)

            if id==1:

                labels.append(0)

            else:

                labels.append(1)



load_images_from_folder_a("../input/cell_images/cell_images/Parasitized",1)

load_images_from_folder_a("../input/cell_images/cell_images/Uninfected/",2)
print(len(images))

print(len(labels))
train = np.array(images)

label=labels

train = train.astype('float32') / 255
if label[1]==1:

    plt.title("Parasitized")

    plt.imshow(train[1])

else:

    plt.title("Uninfected")

    plt.imshow(train[1])

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(train,label,test_size=0.2,random_state=1)
import keras

from keras import Sequential, utils

print(len(X_train),len(X_test))

print(len(Y_train),len(Y_test))

print(X_train[100].shape)

#Doing One hot encoding as classifier has multiple classes

Y_train=keras.utils.to_categorical(Y_train,2)

Y_test=keras.utils.to_categorical(Y_test,2)
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Embedding

from keras.layers import SpatialDropout1D, Conv2D, MaxPooling2D, UpSampling2D

from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(46, 40, 3))) 

model.add(MaxPooling2D((2, 2), padding='same'))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D((2, 2), padding='same'))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dense(128, activation='relu'))

    

model.add(Dense(2, activation='softmax'))



model.summary() 



model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=["accuracy"])


plt.title(Y_train[0])

plt.imshow(X_train[0])
model.fit(X_train,Y_train, epochs=10, batch_size=52, shuffle=True, validation_data=(X_test,Y_test))
accuracy = model.evaluate(X_test, Y_test, verbose=1)

print('\n', 'Test_Accuracy:-', accuracy[1])
from keras.models import load_model

model.save('cells.h5')