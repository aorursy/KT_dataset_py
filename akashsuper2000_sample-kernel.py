import numpy as np

import pandas as pd

import os

from PIL import Image



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score



import keras

import keras.backend as K

from keras.models import Sequential

from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPool2D
train = pd.read_csv('../input/anokha-aiadept-sample/train.csv')

train
data = []

for i in range(1,1501):

    data.append(np.array(Image.open('../input/anokha-aiadept-sample/Train/'+str(i)+'.jpg')))

data = np.array(data)

data.shape

data = data/255

data = data.reshape(1500,64,64,1)

data.shape
y = train.iloc[:,-1].values

y = keras.utils.to_categorical(y)

y.shape
xtrain,xtest,ytrain,ytest = train_test_split(data,y,test_size=0.2)

xtrain.shape,xtest.shape,ytrain.shape,ytest.shape
model = Sequential()



model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=xtrain.shape[1:]))

model.add(Dropout(0.2))

model.add(MaxPool2D((2,2),strides=(2,2)))



model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=xtrain.shape[1:]))

model.add(Dropout(0.2))

model.add(MaxPool2D((2,2),strides=(2,2)))



model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))

model.add(Dropout(0.2))

model.add(MaxPool2D((2,2),strides=(2,2)))



model.add(Flatten())



model.add(Dense(128,activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(32,activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(10,activation='softmax'))



model.summary()



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



model.fit(data,y,epochs=20,validation_split=0.2,batch_size=32)
ypred = model.predict(xtest)

ypred1 = ypred.argmax(axis=1)

ytest1 = ytest.argmax(axis=1)

accuracy_score(ytest1,ypred1)
model.save('model1.h5')
testy = pd.read_csv('../input/anokha-aiadept-sample/test.csv')

testy
testdata = []

for i in range(1,563):

    testdata.append(np.array(Image.open('../input/anokha-aiadept-sample/Test/'+str(i)+'.jpg')))

testdata = np.array(testdata)

testdata.shape

testdata = testdata/255

testdata = testdata.reshape(562,64,64,1)

testdata.shape
ytestpred = model.predict(testdata)

ytestpred = ytestpred.argmax(axis=1)

ytestpred.shape
sample = pd.read_csv('../input/anokha-aiadept-sample/sample.csv')

sample
result = pd.concat([pd.DataFrame(np.arange(1,563)),pd.DataFrame(ytestpred)],axis=1)

result.columns=['id','label']

result
result.to_csv('results.csv',index=False)