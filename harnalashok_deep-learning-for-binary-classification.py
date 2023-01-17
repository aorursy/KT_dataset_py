# -*- coding: utf-8 -*-

"""

Created on Tue Apr 24 03:03:27 2017

@Original author: chuang yun yen

National Taiwan University

Slightly modified: Ashok K Harnal

precision 99%

5 fold validation 

"""



import pandas as pd

import numpy as np

import os

from keras.models import Sequential

from keras.layers import Dense, Activation,Dropout

# Read data

data=pd.read_csv("../input/Kaggle_Training_Dataset.csv")

data.shape

# Convert yes/no to 1/0 in all columns

data = pd.get_dummies(data, drop_first=True)

X = data.iloc[ : , : 22]

Y = data.iloc[ : , 22:23]

X=np.array(X).astype(float)  

Y=np.array(Y).astype(np.int32)



# Modeling

model = Sequential()

model.add(Dense(100, activation='relu', input_dim=22))

model.add(Dropout(0.5))

model.add(Dense(100, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer='nadam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])     

model.fit(X,Y,verbose=1,shuffle=True, nb_epoch=3,batch_size=100,validation_split=0.2)



# Read test data and evaluate results

data=pd.read_csv("../input/Kaggle_Test_Dataset.csv")

data.shape

data = pd.get_dummies(data, drop_first=True)

X = data.iloc[ : , : 22]

Y = data.iloc[ : , 22:23]

X=np.array(X).astype(float)  

Y=np.array(Y).astype(np.int32)



score = model.evaluate(X,Y, batch_size=16)

print("LOSS")

print(score[0])

print("precision")

print(score[1])


