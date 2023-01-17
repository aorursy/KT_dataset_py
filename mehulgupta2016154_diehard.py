# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
x=pd.read_csv('../input/train.csv')
y1=pd.read_csv('../input/test.csv')
x=x.drop(['PassengerId','Name','Ticket','Cabin','Survived'],axis=1)
y1=y1.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

from sklearn.preprocessing import LabelEncoder
x['Sex']=LabelEncoder().fit_transform(x['Sex'])
y1['Sex']=LabelEncoder().fit_transform(y1['Sex'])

x['Embarked']=LabelEncoder().fit_transform(x['Embarked'].astype(str))
y1['Embarked']=LabelEncoder().fit_transform(y1['Embarked'].astype(str))
x=x.dropna(how='any')
y1=y1.dropna(how='any')


bins1=[-1,0,10,25,50,100]
labels1=['invalid','little','teens','adults','old']
x.Age=pd.cut(x.Age,bins=bins1,labels=labels1)
y1.Age=pd.cut(y1.Age,bins=bins1,labels=labels1)


x.Age=LabelEncoder().fit_transform(x.Age)
y1.Age=LabelEncoder().fit_transform(y1.Age)


x['FareCate']=pd.cut(x['Fare'],bins=[0,10,30,50,100,200,300,1000],labels=['lowest','low','medium','high','Higher','very High','Highest'])
y1['FareCate']=pd.cut(y1['Fare'],bins=[0,10,30,50,100,200,300,1000],labels=['lowest','low','medium','high','Higher','very High','Highest'])

x['FareCate']=LabelEncoder().fit_transform(x['FareCate'].astype(str))
y1['FareCate']=LabelEncoder().fit_transform(y1['FareCate'].astype(str))



x=x.drop(['Fare'],axis=1)

y1=y1.drop(['Fare'],axis=1)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
param1={'C':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],'kernel':['linear','rbf']}
param2={'n_neighbors':[3,4,5,6,7,8,9,10]}
param3={'criterion':['gini','entropy']}
param4={'C':[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],'penalty':['l1','l2']}
param5={'learning_rate':[0.1,0.15,0.2,0.3,0.05],'n_estimators':[50,10,100,90,110]}

from sklearn.model_selection import GridSearchCV

model1=GridSearchCV(XGBClassifier(),param5)
x1=pd.read_csv('../input/train.csv')
x1=x1.drop(['Ticket','PassengerId','Cabin','Name'],axis=1)
x1=x1[np.isfinite(x1.Age)]
print(x1.head())
model1.fit(x,x1['Survived'])

print(x.head())
print(y1.head())

print(model1.best_score_)

prediction=model1.predict(y1)
y=pd.read_csv('../input/test.csv')
p=pd.DataFrame(prediction,index=y['PassengerId'],columns=['Survived'])


p.to_csv('mehul3.csv')
