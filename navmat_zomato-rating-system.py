from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt 

import numpy as np 

import os 

import pandas as pd 

from keras.utils import to_categorical

from keras import models

from keras import layers

import math

from sklearn.model_selection import train_test_split

import random
nRowsRead = 50000 # size of dataset

df1 = pd.read_csv('../input/zomato.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'zomato.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')

D = np.array(df1.values)

D = D[:,[5,6,8,9,11,12]]

print(D)



#One-hot encoded location

LDict = {}

cnt = 0

for i in range(len(D[:,2])):

    if(type(D[i,2])==str):

        x = D[i,2]

        if x not in LDict:

            LDict.update({x : cnt})

            cnt = cnt + 1

print(LDict)

for i in range(len(D[:,2])):

    if (type(D[i,2]) == str):

        D[i,2] = LDict[D[i,2]]

    else:

        D[i,2] = random.randint(0,len(LDict)-1)

L_onehot  = to_categorical(D[:,2])

print(L_onehot)
#Restaurant type dictionary

TDict = {}

cnt = 0

for i in range(len(D[:,3])):

    if(type(D[i,3])==str):

        x = D[i,3].split(',')

        #print(x)

        x = np.asarray(x)

        #print(x)

        for j in range(len(x)):

            x[j] = x[j].strip()

            if x[j] not in TDict:

                TDict.update({x[j] : cnt})

                cnt = cnt+1

print(TDict)
#Cost per person

Cost = np.zeros((len(D[:,5]),1))

for i in range(len(D[:,5])):

    if(type(D[i,5]) == str):

        #print(D[i,5])

        D[i,5] = D[i,5].replace(',','')

        if(D[i,5] != 'B'):

            Cost[i] = int(D[i,5])

        else:

            Cost[i] = 0

print(Cost)
#ratings

S = np.zeros((len(D[:,0]),1))

for i in range(len(D[:,0])):

    if(type(D[i,0]) == str):

        x = D[i,0].split('/')

        x = np.asarray(x)

        #print(x)

        if(x[0] != 'NEW' and x[0] != '-'):

            S[i] = float(x[0])

        else:

            S[i] = 0

print(S)
#Cuisine dictionary

CDict = {}

cnt = 0

for i in range(len(D[:,4])):

    if(type(D[i,4])==str):

        x = D[i,4].split(',')

        #print(x)

        x = np.asarray(x)

        #print(x)

        for j in range(len(x)):

            x[j] = x[j].strip()

            if x[j] not in CDict:

                CDict.update({x[j] : cnt})

                cnt = cnt+1

print(CDict)

            

    

#Final Data Entry

print(len(D))

X = np.zeros((len(D),len(LDict) + len(TDict) + len(CDict) + 2))

for i in range(len(X)):

    y = np.zeros(len(CDict))

    t = np.zeros(len(TDict))

    if(type(D[i,4])==str):

        x = D[i,4].split(',')

        #print(x)

        x = np.asarray(x)

        #print(x)

        cnt = 0

        for j in range(len(x)):

            x[j] = x[j].strip()

            val = CDict[x[j]]

            y[val] = 1

            cnt = cnt+1

        #Checking if every restaurant has a valid cuisine

        if(np.sum(y)==0):

            print("Indefinite cuisine")

    if(type(D[i,3])==str):

        x1 = D[i,3].split(',')

        #print(x)

        x1 = np.asarray(x1)

        #print(x)

        cnt = 0

        for j in range(len(x1)):

            x1[j] = x1[j].strip()

            val = TDict[x1[j]]

            t[val] = 1

            cnt = cnt+1    

    #print(len(L_onehot[i]))

    X[i,:] = np.r_[L_onehot[i],t,y,D[i,1],Cost[i]]

print(X)



#Cuisine scores

Y  = np.zeros((len(D),1))

for i in range(len(Y)):

    Y[i] = S[i]

        

print(Y)
#Preparing training and test data

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1)

print(X_train.shape[1])
#Constructing NN model

model = models.Sequential()

model.add(layers.Dense(100, activation = "relu", input_shape=(X_train.shape[1], )))

# Hidden - Layers

model.add(layers.Dense(100, activation = "relu"))

model.add(layers.Dropout(0.3))

model.add(layers.Dense(10, activation = "relu"))

# Output- Layer

model.add(layers.Dense(y_train.shape[1], activation = "linear"))

model.summary()
#Compiling model

model.compile(

 optimizer = "adam",

 loss = "mean_squared_error",

)
#Training model

results = model.fit(

 X_train, y_train,

 epochs= 10,

 batch_size = 100,

 validation_data = (X_test,y_test )

)
