# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import cv2

import os

print(os.listdir("../input"))

datapath='../input'

# Any results you write to the current directory are saved as output.
#How we got Training.npy and other files

# xtrain=[]

# ytrain=[]

# for r,d,f in os.walk(datapath):

#     for x in d:

#         xtrain.append(os.path.join(r,x))

#         ytrain.append(x)





# trainX=[]

# trainY=[]

# for i in xtrain:

#     for r,d,f in os.walk(i):

#         for x in range(3000):

#             trainX.append(os.path.join(r,f[x]))

#             trainY.append(r[len(r)-1])





# data={'FilePath':trainX,'Classes':trainY}

# dataset=pd.DataFrame(data)





# from sklearn.preprocessing import LabelEncoder

# labelencoder=LabelEncoder()

# dataset['Classes']=labelencoder.fit_transform(dataset['Classes'])

# trainY=dataset["Classes"].values





# dim=(32,32)

# img=cv2.imread(trainX[0],cv2.IMREAD_GRAYSCALE)

# img=cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

# Xtrain=np.array([img])

# count=0

# for i in range(1,len(trainX)):

#     count=count+1

#     if(count%100==0):

#         print(count,end=" ")

#     a=cv2.imread(trainX[i],cv2.IMREAD_GRAYSCALE)

# #     print(trainX[i],end="\n")

#     if(a.shape[0]!=32):

#         dim=(32,32)

#         a=cv2.resize(a, dim, interpolation = cv2.INTER_AREA)

    

#     Xtrain=(np.concatenate((Xtrain,np.array([a]))))

#     np.save("Training",Xtrain)

#     np.save("TrainingY",trainY)
xtrain=np.load('../input/Training.npy')

ytrain=np.load('../input/TrainingY.npy')

xtest=np.load('../input/Testing.npy')

ytest=np.load('../input/TestY.npy')
# Showing the samples of the training set take one image from each class  

fig=plt.figure(figsize=(30,12))

m=1

for i in range(1,117000,3000):

    a=fig.add_subplot(3,13,m)

    m=m+1

    plt.imshow(xtrain[i]) 
from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense 
classifier=Sequential()

classifier.add(Convolution2D(32,3,3,input_shape=(32,32,1),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32,3,3,activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim=784,activation='sigmoid'))

classifier.add(Dense(output_dim=784,activation='sigmoid'))

classifier.add(Dense(output_dim=39,activation='softmax'))
classifier.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
xtrain=xtrain.reshape(117000, 32, 32,1)
classifier.fit(xtrain,ytrain,epochs=30,batch_size=200,verbose=1)
xtest=xtest.reshape(117000,32,32,1)
score=classifier.evaluate(xtest,ytest,verbose=1)
print(score)