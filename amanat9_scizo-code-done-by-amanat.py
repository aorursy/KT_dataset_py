# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
import numpy as np

import scipy.misc

import numpy.random as rng

from PIL import Image, ImageDraw, ImageFont

from sklearn.utils import shuffle

import nibabel as nib #reading MR images

from sklearn.model_selection import train_test_split

import math

import glob

from matplotlib import pyplot as plt

%matplotlib inline
gg = glob.glob('../input/dataset/dataset/*')
np.shape(gg)

gg[0]
img ='../input/dataset/dataset/BPDwoPsy_050_MR/anat/NIfTI'
img
ff = glob.glob('../input/dataset/dataset/BPDwoPsy_050_MR/anat/NIfTI/*')
images=[]

labels=[]



for k in gg:

    newstr= str(k)+'/anat/NIfTI/*'

    ff = glob.glob(newstr)

    url = str(k)

    data = url.split("/")



    label = data[-1]

    data = label.split("_")

    real_label=data[0]

    for j in ff:

        

        a = nib.load(ff[0])

        temp=a.shape

#         break

        a = a.get_data()

        a = a[:,:,55:75]

        

        for i in range(a.shape[2]):

            labels.append(real_label)

            images.append((a[:,:,i]))

        print (a.shape)

# print(temp)
np.shape(images)
np.shape(labels)
images = np.asarray(images)

images = images.reshape(-1, 256,256,1)
labels
myset = set(labels)

myset
labels.count('BPDwoPsy')
labels=np.asarray(labels)
labels[labels=='BPDwPsy']=0

labels[labels=='BPDwoPsy']=0

labels[labels=='HC']=1

labels[labels=='SS']=1
import tensorflow as tf

from tensorflow.keras.datasets import cifar10

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from tensorflow.keras.layers import Conv2D, MaxPooling2D
y = labels

X = images/255.0



model = Sequential()



model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(256, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors



model.add(Dense(64))



model.add(Dense(3))

model.add(Activation('sigmoid'))



model.compile(loss='sparse_categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



model.fit(X, y, batch_size=32, epochs=15, validation_split=0.1)
model.fit(X, y, batch_size=32, epochs=3, validation_split=0.1)
print(images[0].shape)