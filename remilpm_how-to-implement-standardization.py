import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

Data_Set1= pd.read_csv("../input/Titanic.csv")

Data_Set1.tail()

#https://www.kaggle.com/rakend/multiple-linear-regression-with-gradient-descent
Data_Set1.isnull().dropna

cond=(Data_Set1['Age']>0)

Data_Set1=Data_Set1[cond]

x=Data_Set1[['Age','SibSp','Parch','Fare']]

y=Data_Set1['Survived']

x.head()

#Setup train and validation cross validation datasets

from sklearn.model_selection import train_test_split

X_train,X_validation,Y_train,Y_validation=train_test_split(x,y,test_size=0.3,random_state=0)

X_train.shape,X_validation.shape,Y_train.shape,Y_validation.shape
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,Y_train)
from sklearn.metrics import accuracy_score

y_pred=lr.predict(X_validation)

print(accuracy_score(y_pred,Y_validation))
#Standardization

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_std=sc.fit_transform(X_train)

X_std

X_validation_std=sc.fit_transform(X_validation)

X_validation_std
lr1 = LogisticRegression()

lr1.fit(X_std,Y_train)
from sklearn.metrics import accuracy_score

y_pred1=lr1.predict(X_validation_std)

print(accuracy_score(y_pred1,Y_validation))