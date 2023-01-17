import numpy as np

import pandas as pd

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras.preprocessing import image

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from tqdm import tqdm

import os

import cv2
train_path='../input/train/train'

test_path='../input/test/test'
train = pd.read_csv("../input/train.csv")

train.head()
id=train['id'].values

cactus=train['has_cactus'].values



train=[]

X=[]

Y=[]

a=0



for i in tqdm(sorted(os.listdir(train_path))):

    path=os.path.join(train_path,i)

    i=cv2.imread(path,cv2.IMREAD_COLOR)

    X.append(i)

    train.append([np.array(cactus),cactus[a]])

    a=a+1



train=np.array(train)

Y=train[:,1]

train=train[:,0]

X=np.array(X)



X.shape



X=X/255

train=train/255
test1=[]

X_test=[]



for i in tqdm(os.listdir(test_path)):

    id=i

    path=os.path.join(test_path,i)

    i=cv2.imread(path,cv2.IMREAD_COLOR)

    X_test.append(i)

    test1.append([np.array(i),id])



X_test=np.array(X_test)

X_test.shape

test1=np.array(test1)

id_test=test1[:,1]

test1=test1[:,0]

test1.shape



X_test=X_test/255

test1=test1/255
model = Sequential()

model.add(Conv2D(filters=128,kernel_size=2,padding="same",activation="relu",input_shape=(32,32,3)))

model.add(MaxPooling2D(pool_size=2,strides=1))

model.add(Dropout(0.2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2,strides=1))

model.add(Dropout(0.2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2,strides=1))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(32,activation="relu"))

model.add(Dropout(0.7))

model.add(Dense(1,activation="sigmoid"))

model.summary()
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

h=model.fit(X,Y,batch_size=256,validation_split=0.2,epochs=100)
pred=model.predict(X_test)

ids=[]

label=[]

a=0

for i in tqdm(os.listdir(test_path)):

    id=i

    ids.append(id)

    label.append(pred[a])

    a=a+1



label=np.array(label,dtype='float64')

out=pd.DataFrame({'id': ids,'has_cactus':label[:,0]})



out.to_csv('submission.csv',index=False,header=True)