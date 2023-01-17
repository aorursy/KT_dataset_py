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
train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
train.shape
test.head()
test.shape
train.isnull().sum()
train['Age']=train['Age'].fillna(train['Age'].mean())
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import math
train['Embarked'].value_counts()
train['Embarked']=train['Embarked'].fillna('S')

train=train.drop(['Cabin'], axis=1)

sns.heatmap(train.isnull())
pd.get_dummies(train['Sex'])

sex=pd.get_dummies(train['Sex'],drop_first=True)

embarked=pd.get_dummies(train['Embarked'])
dd=pd.concat([train,sex,embarked],axis=1)

dd.head()
d1=dd.drop(['Name','Sex','Ticket','Embarked','PassengerId'],axis=1)
d1.head()
X=d1.drop('Survived',axis=1)

y=d1['Survived']
X.head()
y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
lm=LogisticRegression()

lm.fit(X_train,y_train)
pred=lm.predict(X_test)
from sklearn.metrics import classification_report
classification_report(y_test,pred)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred)