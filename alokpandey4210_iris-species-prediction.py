# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/iris/Iris.csv')

df.head()
df['Species'].value_counts()
df['Species']=df['Species'].map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})
df.corr(method='pearson')
X=df.drop(['Id','Species'],axis=1)

y=df['Species']
X.head()
X.isnull().sum()
X=pd.get_dummies(X)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.40)
from sklearn.linear_model import LogisticRegression

s=LogisticRegression()

s.fit(X_train,y_train)

s.score(X_test,y_test)
from sklearn.svm import SVC

qw=SVC()

qw.fit(X_train,y_train)

qw.score(X_test,y_test)
from sklearn.neighbors import KNeighborsClassifier

ds=KNeighborsClassifier()

ds.fit(X_train,y_train)

ds.score(X_test,y_test)
from sklearn.naive_bayes import GaussianNB

ih=GaussianNB()

ih.fit(X_train,y_train)

ih.score(X_test,y_test)
from sklearn.linear_model import SGDClassifier

kd=SGDClassifier()

kd.fit(X_train,y_train)

kd.score(X_test,y_test)
from sklearn.tree import DecisionTreeClassifier

ls=DecisionTreeClassifier()

ls.fit(X_train,y_train)

ls.score(X_test,y_test)

from sklearn.ensemble import RandomForestClassifier

oa=RandomForestClassifier()

oa.fit(X_train,y_train)

oa.score(X_test,y_test)
from sklearn.metrics import confusion_matrix

y_predict=ds.predict(X_test)

results=confusion_matrix(y_test,y_predict)

print("confusion matrix")

print(results)

# saving model

import pickle 

file_name='Iris.sav'

tuples=(ds,X)

pickle.dump(tuples,open(file_name,'wb'))
from sklearn.metrics import confusion_matrix

y_predict1=ih.predict(X_test)

results1=confusion_matrix(y_test,y_predict1)



print(results1)

from sklearn.metrics import confusion_matrix

y_predict2=kd.predict(X_test)

results2=confusion_matrix(y_test,y_predict2)



print(results2)

from sklearn.metrics import confusion_matrix

y_predict3=ls.predict(X_test)

results3=confusion_matrix(y_test,y_predict3)



print(results3)

from sklearn.metrics import confusion_matrix

y_predict4=oa.predict(X_test)

results4=confusion_matrix(y_test,y_predict4)



print(results4)

yya1=s.predict_proba(X_test)



ya3=ds.predict_proba(X_test)

ya4=ih.predict_proba(X_test)



ya6=ls.predict_proba(X_test)

ya7=oa.predict_proba(X_test)

print(yya1)

print(ya3)

print(ya4)

print(ya6)

print(ya7)