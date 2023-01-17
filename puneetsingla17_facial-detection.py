import numpy as np # linear algebra

import pandas as pd # data processing, 

import tensorflow as tf

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))

import glob

from tqdm import tqdm

import cv2

from PIL import ImageDraw,Image

traindf=pd.read_csv('../input/training/training.csv')

testdf=pd.read_csv('../input/test/test.csv')

from keras.layers.advanced_activations import LeakyReLU
columns=[i for i in traindf.columns if 'eye_center' in i or 'nose' in i or'bottom_lip' in i]
traindf.info()
traindf.shape
X_train=np.stack([np.array(list(map(int,i.split(" ")))).reshape((96,96)) for i in traindf.Image])

X_test=np.stack([np.array(list(map(int,i.split(" ")))).reshape((96,96)) for i in testdf.Image])



img=X_train[16].astype(np.float32)

m=traindf.loc[2,columns]
m
a=Image.fromarray(img)

draw=ImageDraw.Draw(a)

plt.imshow(a)
for i in range(len(m)//2):

    if i==len(m)-1:

        continue

    draw.ellipse((m[2*i]-1,m[2*i+1]-1,m[2*i]+1,m[2*i+1]+1),fill=255)
plt.imshow(a)
X_train1=X_train.astype(np.float32)/255
Ytrain=traindf.loc[:,columns]

    

X_train1.shape
indexes=Ytrain[pd.isna(Ytrain).any(axis=1)].index

X=np.delete(X_train1,indexes,axis=0)
X.shape
Y=Ytrain.dropna(axis=0)

Y.shape
Y=Y.values
Y=Y.astype(np.float32)
Y.shape
X.shape
X=X.reshape((-1,96,96,1))
X.shape
#Making Model

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Activation,BatchNormalization,InputLayer

from keras.models import Model,Sequential
model=Sequential()

model.add(InputLayer((96,96,1)))

model.add(Conv2D(16,kernel_size=(3,3),padding='same'))

model.add(BatchNormalization())

model.add(Activation(LeakyReLU(0.2)))

model.add(MaxPooling2D())

model.add(Conv2D(32,kernel_size=(3,3),padding='same'))

model.add(BatchNormalization())

model.add(Activation(LeakyReLU(0.2)))

model.add(MaxPooling2D())

model.add(Conv2D(32,kernel_size=(3,3),padding='same'))

model.add(BatchNormalization())

model.add(Activation(LeakyReLU(0.2)))

model.add(MaxPooling2D())

model.add(Conv2D(64,kernel_size=(3,3),padding='same'))

model.add(BatchNormalization())

model.add(Activation(LeakyReLU(0.2)))

model.add(MaxPooling2D())

model.add(Conv2D(128,kernel_size=(3,3),padding='same'))

model.add(BatchNormalization())

model.add(Activation(LeakyReLU(0.2)))

model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(8))                     # Mistake fixed removed softmax since we are not normalizing the probability its a regression 

model.compile(loss='mse',optimizer='adam')
model.summary()
history = model.fit(X,Y,epochs=100,batch_size=32)
Xtest1=X_test[0:10].astype(np.float32)/255

Xtest1=Xtest1.reshape((-1,96,96,1))
ypred=model.predict(Xtest1)
ypred
m=ypred[0]

xtest1=Xtest1[0].astype(np.float32)*255.0       # have to multiply image with 255 otherwise image is shown to be black

xtest1=xtest1.reshape((96,96))
a=Image.fromarray(xtest1)

draw=ImageDraw.Draw(a)
plt.imshow(xtest1)
for i in range(len(m)//2):

    if i==len(m)-1:

        continue

    draw.ellipse((m[2*i]-1,m[2*i+1]-1,m[2*i]+1,m[2*i+1]+1),fill=255)
plt.imshow(a)  # with MSE  worked good
m=ypred[1]

xtest1=Xtest1[1].astype(np.float32)*255.0       # have to multiply image with 255 otherwise image is shown to be black

xtest1=xtest1.reshape((96,96))

a=Image.fromarray(xtest1)

draw=ImageDraw.Draw(a)

plt.imshow(xtest1)
for i in range(len(m)//2):

    if i==len(m)-1:

        continue

    draw.ellipse((m[2*i]-1,m[2*i+1]-1,m[2*i]+1,m[2*i+1]+1),fill=255)

    

plt.imshow(a) 
traindf.info()
columns=[i for i in traindf.columns if 'eyebrow' in i or 'corner' in i or 'top_lip' in i]
columns
#Eye /eyebrow /mouth classifier

Y=traindf.loc[:,columns]
indexes=Y[pd.isna(Y).any(axis=1)].index
Xtrain3=np.delete(X_train1,indexes,axis=0)
Xtrain3.shape
Y=Y.dropna(axis=0)
Y.shape
x1=Xtrain3[1]*255

m=Y.iloc[1,:]
plt.imshow(x1)
x1=Image.fromarray(x1)

draw=ImageDraw.Draw(x1)

for i in range(len(m)//2):

    if i==len(m)-1:

        continue

    draw.ellipse((m[2*i]-1,m[2*i+1]-1,m[2*i]+1,m[2*i+1]+1),fill=255)
plt.imshow(x1)
Xtrain3=Xtrain3.astype(np.float32)
Xtrain3.shape
Xtrain3=Xtrain3.reshape((-1,96,96,1))
Y=Y.values

Y=Y.astype(np.float32)
Y.shape
model1=Sequential()

model1.add(InputLayer((96,96,1)))

model1.add(Conv2D(16,kernel_size=(3,3),padding='same'))

model1.add(BatchNormalization())

model1.add(Activation(LeakyReLU(0.2)))

model1.add(MaxPooling2D())

model1.add(Conv2D(32,kernel_size=(3,3),padding='same'))

model1.add(BatchNormalization())

model1.add(Activation(LeakyReLU(0.2)))

model1.add(MaxPooling2D())

model1.add(Conv2D(32,kernel_size=(3,3),padding='same'))

model1.add(BatchNormalization())

model1.add(Activation(LeakyReLU(0.2)))

model1.add(MaxPooling2D())

model1.add(Conv2D(64,kernel_size=(3,3),padding='same'))

model1.add(BatchNormalization())

model1.add(Activation(LeakyReLU(0.2)))

model1.add(MaxPooling2D())

model1.add(Conv2D(128,kernel_size=(3,3),padding='same'))

model1.add(BatchNormalization())

model1.add(Activation(LeakyReLU(0.2)))

model1.add(MaxPooling2D())

model1.add(Flatten())

model1.add(Dense(22))                     # Mistake fixed removed softmax since we are not normalizing the probability its a regression 

model1.compile(loss='mse',optimizer='adam')
model1.summary()
history = model1.fit(Xtrain3,Y,epochs=100,batch_size=32)
Xtest1.shape
ypred=model1.predict(Xtest1)
x=Xtest1[5].reshape((96,96))*255
plt.imshow(x)
m=ypred[5]
m
x1=Image.fromarray(x)
draw=ImageDraw.Draw(x1)
for i in range(len(m)//2):

    if i==len(m)-1:

        continue

    draw.ellipse((m[2*i]-1,m[2*i+1]-1,m[2*i]+1,m[2*i+1]+1),fill=255)

    

plt.imshow(x1) 
df=pd.read_csv('../input/IdLookupTable.csv')
df
traindf.columns