import requests

from bs4 import BeautifulSoup

import lxml

import os

import urllib

import sys

import pandas as pd

import numpy as np

from PIL import Image

import cv2

import csv

import multiprocessing

import matplotlib.pyplot as plt
os.listdir('../input/iwildcam-2019-fgvc6')
train=pd.read_csv("../input/iwildcam-2019-fgvc6/train.csv")

test=pd.read_csv("../input/iwildcam-2019-fgvc6/test.csv")

train.head()
len(os.listdir('../input/iwildcam-2019-fgvc6/train_images'))
len(train.id)
img=[]

filename=train.id[:10000]

label=train.category_id[:10000]

for file in filename:

    image=cv2.imread("../input/iwildcam-2019-fgvc6/train_images/"+file+'.jpg')

    res=cv2.resize(image,(32,32))

    img.append(res)

img=np.array(img)
plt.figure(figsize=(15,15))

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.imshow(img[i])
np.random.seed(921)

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Convolution2D,Dense,MaxPool2D,Activation,Dropout,Flatten

from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from keras.layers.normalization import BatchNormalization

X_train,X_test,y_train,y_test=train_test_split(img,label,test_size=0.2)

del img

y_train=y_train.astype(int)

y_test=y_test.astype(int)

y_train=np.array(y_train).reshape(-1,1)

y_test=np.array(y_test).reshape(-1,1)

X_train=X_train.reshape(-1,32,32,3)/255 #Normalize

X_test=X_test.reshape(-1,32,32,3)/255

y_train=np_utils.to_categorical(y_train,num_classes=max(label)+1)

y_test=np_utils.to_categorical(y_test,num_classes=max(label)+1)
model=Sequential()

model.add(Convolution2D(filters=32,kernel_size=(3,3),input_shape=(32,32,3),padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(rate=0.35))



model.add(MaxPool2D(pool_size=(2,2),padding='same'))



model.add(Convolution2D(filters=64,kernel_size=(3,3),padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(rate=0.45))



model.add(MaxPool2D(pool_size=(2,2),padding='same'))



model.add(Flatten())



model.add(Dense(1024,activation='relu'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(rate=0.75))



model.add(Dense(max(label)+1,activation='softmax'))



model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])



train_history=model.fit(X_train,y_train,validation_split=0.2,epochs=20,batch_size=128,verbose=1)

accuracy=model.evaluate(X_test,y_test,verbose=1)

print("test accuracy:",accuracy[1])#accuracy for test set







def show_train_history(train_history,train,validation):

	plt.plot(train_history.history[train])

	plt.plot(train_history.history[validation])

	plt.title('Train History')

	plt.ylabel('train')

	plt.xlabel('Epoch')

	plt.legend(['train','validation'],loc='upper left')

	plt.show()



show_train_history(train_history,'acc','val_acc') #acc:accuracy for training set. val_acc:accuracy for validation.
prediction=model.predict_classes(X_test)

print(prediction[0:10])
img_test=[]

filename_test=test.id[:10000]

for file in filename_test:

    image=cv2.imread("../input/iwildcam-2019-fgvc6/test_images/"+file+'.jpg')

    res=cv2.resize(image,(32,32))

    img_test.append(res)

img_test=np.array(img_test)
prediction=model.predict_classes(img_test)

print(prediction[0:10])
submit=pd.DataFrame({'Id':filename_test,'Predicted':prediction})

submit.to_csv('submission.csv',index=False)