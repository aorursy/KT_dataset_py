import numpy as np

import matplotlib.pyplot as plt

import os

from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense, Activation
data = np.loadtxt('../input/digit-recognizer/train.csv',skiprows=1,delimiter=',')
data.shape
trainLabel = data[:,0].reshape(-1,1)
trainData = data[:,1:].astype(float)/255.
enc = OneHotEncoder()

enc.fit(trainLabel)

trainLabelHot = enc.transform(trainLabel[:,:]).toarray()
model = Sequential()

model.add(Dense(units=128,input_dim=trainData.shape[1],activation='relu'))

model.add(Activation('relu'))

model.add(Dense(units=32,input_dim=trainData.shape[1],activation='relu'))

model.add(Activation('relu'))

model.add(Dense(units=10))

model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',

              optimizer='sgd',

              metrics=['accuracy'])
model.fit(trainData,trainLabelHot,batch_size=128,epochs=20)
N=10000

start = 10000

selected_to_test = slice(start,start+N)

dataPredictions = model.predict(trainData[selected_to_test,:])
predLabels = dataPredictions.argmax(axis=1)

predOuts = dataPredictions[range(N),predLabels]

good_predictions = predLabels==trainLabel.flatten().astype(int)[selected_to_test]

plt.hist(predOuts[good_predictions],bins=25,color='g',alpha=0.5)

plt.hist(predOuts[np.logical_not(good_predictions)],bins=25,color='r',alpha=0.5)
nfigures = 10

error_arr = np.zeros((nfigures,nfigures),dtype=float)

for idx in np.flatnonzero(np.logical_not(good_predictions)):

    error_arr[predLabels[idx],int(trainLabel[idx][0])]+=1

for x in range(nfigures):

    for y in range(nfigures):

        plt.text(s=str(error_arr[x,y]),x=x,y=y)

plt.pcolormesh(((error_arr)/N).T)

plt.xlabel('Predicted')

plt.ylabel('Ground truth')

plt.colorbar()
print(np.sum(good_predictions)/N)