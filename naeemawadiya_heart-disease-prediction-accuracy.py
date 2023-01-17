import numpy as np

import pandas as pd

import seaborn as sb

import matplotlib as plt

%matplotlib inline

import os as os
os.chdir("../input")
os.listdir()
data=pd.read_csv('../input/heart.csv')
data.head(3)
data.shape
from sklearn.model_selection import train_test_split
train,test=train_test_split(data,test_size=0.2)
len(train)
len(test)
data.corr()
sb.scatterplot(x=train['cp'], y= train['age'],hue=train['target'],data=train)
sb.countplot(x=train['cp'],hue=train['target'],data=train)
train.isnull().sum()
sb.heatmap(train.isnull(),yticklabels=False,cbar='False')
train.describe()
test.isnull().sum()
sb.boxplot(x='age',y='target',data=train,palette='Set1')
train_out=train['target']
train_in=train.drop(['target'],axis=1)
train_in.shape
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()

log_reg.fit(train_in,train_out)
test_out=test['target']
test_in=test.drop(['target'],axis=1)
prediction=log_reg.predict(test_in)
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
accuracy_score(test_out,prediction)*100
confusion_matrix(test_out,prediction)
print(classification_report(test_out,prediction))