# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/creditcard.csv',header=0)

data.head()
print(data.shape)

print('+'*50)

#print(data.info())

data.info()
fig=data.plot(kind='Line',x='Time',y='Amount')

fig=plt.gcf()

fig.set_size_inches(8,6)

plt.show()
fig=data.plot(kind='hist',x='Amount',y='Class',bins=2,rwidth=0.8)

fig=plt.gcf()

fig.set_size_inches(8,6)

plt.show()
fig=data.plot(kind='scatter',x='Class',y='Amount')

fig=plt.gcf()

fig.set_size_inches(8,6)

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm

from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
data.corr()
plt.figure(figsize=(25,20))

sns.heatmap(data.corr(),annot=True)

plt.show()
cordata=data[['Time','V7','V20','Amount','Class']]

cordata.head()
plt.figure(figsize=(15,10))

sns.heatmap(cordata.corr(),annot=True)

plt.show()
train,test=train_test_split(data,test_size=0.3,random_state=0)

print(train.shape)

print('-'*70)

print(test.shape)

print('-'*70)

print(train.head())

print('-'*70)

print(test.head())
x_train=train[['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',

              'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',

             'V21','V22','V23','V24','V25','V26','V27','V28','Amount']]

y_train=train[['Class']]

x_test=test[['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',

              'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',

             'V21','V22','V23','V24','V25','V26','V27','V28','Amount']]

y_test=test[['Class']]
model=LogisticRegression()

model.fit(x_train,y_train)

prediction1=model.predict(x_test)

print('The accuracy of Logistic Regression is :',metrics.accuracy_score(prediction1,y_test)*100)