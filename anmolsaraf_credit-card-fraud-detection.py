# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier

import random
creditcard = pd.read_csv("../input/creditcardfraud/creditcard.csv")

creditcard.head()

print("Shape of the dataframe: ", creditcard.shape)
creditcard_array=creditcard.values

print(type(creditcard_array))

np.random.shuffle(creditcard_array)

print(type(creditcard_array))

creditcard_array[:4,:10]

X_raw=creditcard_array[:,:30]

minmaxscaler=MinMaxScaler()

X=minmaxscaler.fit_transform(X_raw)

Y=creditcard_array[:,-1]

X.shape

Y.shape
X_train,X_test, Y_train,Y_test=train_test_split(X,Y, test_size=0.2)

model_KNN=KNeighborsClassifier(n_neighbors=2)

model_KNN.fit(X_train, Y_train)

Y_pred=model_KNN.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(Y_test,Y_pred)

#accuracy=100*np.sum(np.multiply(Y_pred, Y_test))/len(Y_test)

print("Accuracy= ",accuracy)
from sklearn.linear_model import LogisticRegression

model_LR=LogisticRegression()

model_LR.fit(X_train, Y_train)

Y_pred=model_LR.predict(X_test)

accuracy=accuracy_score(Y_test,Y_pred)

print("Accuracy= ",accuracy)
from keras.models import Sequential

from keras.layers import Dense

import tensorflow as tf
model_DL = Sequential()

model_DL.add(Dense(40,activation='relu', input_shape=(30,)))

model_DL.add(Dense(65, activation='relu'))

model_DL.add(Dense(30,activation='relu'))

model_DL.add(Dense(1, activation='sigmoid'))

model_DL.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',])

model_DL.fit(X_train,Y_train, batch_size=20000, epochs=5)
model_DL.evaluate(X_test, Y_test)
i=np.where(Y_train==0)

print('example=',Y_test[2950])

print('example_predicted=', np.mean(model_DL.predict(X_train[i])))
from sklearn.naive_bayes import GaussianNB

model_GNB = GaussianNB()

model_GNB.fit(X_train, Y_train)

Y_pred=model_GNB.predict(X_test)

accuracy=accuracy_score(Y_test,Y_pred)

print("Accuracy= ",accuracy)