# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import cv2
import random

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
data = '../input/flowers/flowers/'
folders = os.listdir('../input/flowers/flowers/')
print(folders)
# Any results you write to the current directory are saved as output.
import cv2

img_names=[]
train_labels=[]
train_images=[]

size=64,64

for folder in folders :
    for file in os.listdir(os.path.join(data,folder)) :
        if(file.endswith('jpg')) :
            img_names.append(os.path.join(data,folder,file))
            train_labels.append(folder)
            img = cv2.imread(os.path.join(data,folder,file))
            im = cv2.resize(img,size)
            train_images.append(im)
train = np.array(train_images)
train.shape
train = train.astype('float32')/255.0
label_dummies = pd.get_dummies(train_labels)
print(label_dummies)
labels = label_dummies.values.argmax(1)
pd.unique(train_labels)

pd.unique(labels)
num_classes=5
union_list=list(zip(train,labels))
random.shuffle(union_list)
train,labels = zip(*union_list)
train=np.array(train)
labels=np.array(labels)
out_y=keras.utils.to_categorical(labels, num_classes)
from tensorflow.python.keras.layers import Dense, Flatten,Conv2D,Dropout, GlobalAveragePooling2D
img_rows,img_cols = 64,64

model = keras.Sequential()
model.add(Conv2D(20, kernel_size=(3, 3),
                 strides=2,
                 activation='relu',
                 input_shape=(img_rows, img_cols, 3)))
model.add(Dropout(0.5))
model.add(Conv2D(20, kernel_size=(3, 3), strides=2, activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(20, kernel_size=(3, 3), strides=2, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
model.fit(train, out_y,batch_size=128,epochs=6,validation_split = 0.2)
