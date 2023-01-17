# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import cv2

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%ls '../input/'

labels=pd.read_csv('../input/labels.csv')

test=pd.read_csv('../input/sample_submission.csv')
path=labels['id']

target1=labels['breed']

testpath=test['id']
target=target1.append(target1)

target=target.append(target)
from keras.preprocessing.image import load_img,img_to_array

image_train1=[]

for x in path:

    y=cv2.imread('../input/train/'+x+'.jpg')

    y=cv2.resize(y,(64,64))

    y=np.array(y).flatten()

    image_train1.append(y)
image_test=[]

for x in testpath:

    y=cv2.imread('../input/test/'+x+'.jpg')

    y=cv2.resize(y,(64,64))

    y=np.array(y).flatten()

    image_test.append(y)
image_train=np.array(image_train1)

image_test=np.array(image_test)
image_train=np.append(image_train,image_train,0)

image_train=np.append(image_train,image_train,0)
target_all=pd.get_dummies(target,sparse=True)

target=np.asarray(target_all)
from keras.layers import Dense,Dropout,Flatten

from sklearn.model_selection import train_test_split as tts

xtrain,xtest,ztrain,ztest=tts(image_train,target,train_size=0.8)
from keras.layers import Conv2D

from keras.models import Sequential

model=Sequential()

model.add(Dense(1000, activation = "tanh"))

model.add(Dropout(0.1))

model.add(Dense(500, activation = "relu"))

model.add(Dropout(0.1))

model.add(Dense(120, activation = "softmax"))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history=model.fit(xtrain,ztrain, batch_size=5000, epochs=10)