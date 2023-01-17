# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from matplotlib import pyplot as plt

from keras.preprocessing.image import load_img

from tqdm import tqdm

from keras.models import Sequential

from keras.layers import Convolution2D,BatchNormalization,Dropout,Dense,Flatten,MaxPool2D

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from sklearn.metrics import classification_report

# Any results you write to the current directory are saved as output.
uninfectedBaseDir='../input/cell_images/cell_images/Uninfected/'

infectedBaseDir='../input/cell_images/cell_images/Parasitized/'

validImageFormats=['tif','jpg','png','bmp']

nonInfectedImgFileList=[x for x in os.listdir(uninfectedBaseDir) if x.split('.')[1] in validImageFormats]

infectedImgFileList=[x for x in os.listdir(infectedBaseDir) if x.split('.')[1] in validImageFormats]
nonInfectedImagesList=[]

infectedImagesList=[]

for nf in tqdm(nonInfectedImgFileList):

    nonInfectedImagesList.append([load_img(path=uninfectedBaseDir+nf,target_size=(100,100)),0])

print('Loaded Non Infected Images')

for pf in tqdm(infectedImgFileList):

    infectedImagesList.append([load_img(path=infectedBaseDir+pf,target_size=(100,100)),1])

print('Loaded infected Images')
len(nonInfectedImagesList),len(infectedImagesList)
nX=[np.array(x[0]) for x in nonInfectedImagesList]

ny=[x[1] for x in nonInfectedImagesList]

pX=[np.array(x[0]) for x in infectedImagesList]

py=[x[1] for x in infectedImagesList]
X=np.vstack((nX,pX))

y=np.array(ny+py)

#y=y.reshape(y.shape[0],1)

num_classes=2

y=to_categorical(num_classes=num_classes,y=y)

input_shape=(100,100,3)
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.80,test_size=0.20,random_state=43,shuffle=True)
X_train.shape,y_train.shape,X_test.shape,y_test.shape
def createCNNModel():

    model=Sequential()

    model.add(Convolution2D(64,3,input_shape=input_shape))

    model.add(Convolution2D(64,3,activation='relu'))

    model.add(MaxPool2D(2))

    model.add(BatchNormalization())

    

    model.add(Convolution2D(128,3,activation='relu'))

    model.add(Convolution2D(128,3,activation='relu'))

    model.add(MaxPool2D(2))

    model.add(BatchNormalization())

    

    model.add(Convolution2D(128,3,activation='relu'))

    model.add(Convolution2D(128,3,activation='relu'))

    model.add(MaxPool2D(2))

    model.add(BatchNormalization())

    

    model.add(Convolution2D(256,3,activation='relu'))

    model.add(Convolution2D(256,3,activation='relu'))

    model.add(MaxPool2D(2))

    model.add(BatchNormalization())

    

    model.add(Flatten())

    model.add(Dense(500,activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(100,activation='relu'))

    model.add(Dropout(0.2))

    

    model.add(Dense(num_classes,activation='sigmoid'))

    return model
model=createCNNModel()

model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(X_train,y_train,epochs=10,batch_size=128,verbose=1)
model.evaluate(X_test,y_test)[1]*100
rndIdx=np.random.randint(len(X_test)-1)

plt.imshow(X_test[rndIdx])

print('Predicted:',np.argmax(model.predict(X_test[rndIdx].reshape(1,100,100,3))))

print('Actual:',np.argmax(y_test[rndIdx]))