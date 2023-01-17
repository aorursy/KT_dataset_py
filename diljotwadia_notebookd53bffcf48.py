import pandas as pd

from sklearn.preprocessing import OneHotEncoder

import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow import keras
dataset = pd.read_csv(r'../input/cat-in-the-dat/train.csv')

testset = pd.read_csv(r'../input/cat-in-the-dat/test.csv')
X= dataset.loc[:,:'month']

Y= dataset.loc[:,'target']
alldata = pd.concat((X,testset))

alldata.drop('id', axis=1, inplace=True)
print(str(X.shape[0])+" rows of X")

print(str(testset.shape[0])+" rows of testSet")

print(str(alldata.shape[0])+" rows of Combined")
ohcInstance=OneHotEncoder()

ohcInstance.fit(alldata)

alldata=ohcInstance.transform(alldata)
print("After one hot encoding no. of columns become "+str(alldata.shape[1]))
X=alldata[0:300000]

Test_X=alldata[300000:]
model = keras.models.Sequential([

                                 tf.keras.layers.Dense(512,input_dim=X.shape[1],activation='relu'),

                                tf.keras.layers.Dense(128,activation='relu'),

                                 tf.keras.layers.Dense(64,activation='relu'),

                                  tf.keras.layers.Dense(1, activation='sigmoid')

])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()
history_OneHot=model.fit(X, np.asarray(Y).astype(np.int32), epochs=11, batch_size=10000, verbose=1)

predictions = model.predict(Test_X)

submit = pd.concat([testset['id'], pd.Series(predictions[:,0]).rename('target')], axis=1)

submit.to_csv('submission.csv', index=False, header=True)
submit
dataset = pd.read_csv(r'../input/cat-in-the-dat/train.csv')



X= dataset.iloc[:,1:24].values

Y= dataset.iloc[:,24].values

def labelEncode(listData,index):

    labelEncoder=LabelEncoder()

    listData[:,index]=labelEncoder.fit_transform(listData[:,index]) 
dataset.iloc[:,1:24]
from sklearn.preprocessing import LabelEncoder

for i in range(3,15):

    labelEncode(X,i)

for i in range(16,21):

    labelEncode(X,i)

model = keras.models.Sequential([

                                 tf.keras.layers.Dense(512,input_dim=X.shape[1],activation='relu'),

                                tf.keras.layers.Dense(128,activation='relu'),

                                 tf.keras.layers.Dense(64,activation='relu'),

                                  tf.keras.layers.Dense(1, activation='sigmoid')

])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
history_label=model.fit(np.asarray(X).astype(np.int32), np.asarray(Y).astype(np.int32), epochs=11, batch_size=10000, verbose=1)
# PLOT LOSS AND ACCURACY

%matplotlib inline



import matplotlib.image  as mpimg

import matplotlib.pyplot as plt

oneacc=history_OneHot.history['acc']

labelacc=history_label.history['acc']

oneloss=history_OneHot.history['loss']

labelloss=history_label.history['loss']
epochs=range(len(oneacc)) # Get number of epochs



plt.plot(epochs, oneacc, 'b', "Label Encoding Accuracy")

plt.plot(epochs, labelacc, 'r', "OneHotEncoding Accuracy")

plt.title('Difference between accuracy of Label and OneHotEncoder')

plt.figure()





plt.plot(epochs, oneloss, 'r', "Label Encoding  Loss")

plt.plot(epochs, labelloss, 'b', "OneHotEncoding Loss")

plt.title('Difference between Loss of Label and OneHotEncoder')

plt.figure()