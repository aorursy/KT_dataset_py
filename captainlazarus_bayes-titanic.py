import numpy as np

import pandas as pd

import os

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score 

from sklearn.metrics import confusion_matrix



trainData = pd.read_csv("/kaggle/input/titanic/train.csv")

testData = pd.read_csv("/kaggle/input/titanic/test.csv")
X = trainData.drop(['Survived','Name','Cabin',"PassengerId","Ticket"],axis = 1)

Y = trainData['Survived']



X = X.dropna(axis=1)

new_X = pd.get_dummies(X) #OneHot Encoding





X_train , X_test , Y_train , Y_test = train_test_split(new_X,Y,test_size=0.2 , random_state=0)
g = GaussianNB()

model = g.fit(X_train , Y_train)

p = model.predict(X_test)

acc = accuracy_score(p,Y_test)



print(acc)