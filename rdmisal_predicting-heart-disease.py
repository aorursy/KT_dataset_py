import numpy as np 

import pandas as pd 

import os

data=pd.read_csv('../input/heart.csv')

data.head()
data.describe()
data.hist(figsize=(20,10))
data.corr()
from sklearn.model_selection import train_test_split

train=data.drop('target',axis=1)

test=data['target']

from sklearn.preprocessing  import StandardScaler

scaler=StandardScaler()

scaler.fit(train)

x_train,x_test,y_train,y_test=train_test_split(train,test,test_size=0.33)
from sklearn.naive_bayes import BernoulliNB

model=BernoulliNB()

model.fit(x_train,y_train)

model.predict(x_test)
from sklearn.metrics import accuracy_score, confusion_matrix

predict=model.predict(x_test)

print("Accuracy score is ", accuracy_score(y_test,predict))
confusion_matrix(y_test,predict)
from sklearn.metrics import precision_score, classification_report

classification_report(y_test,predict)