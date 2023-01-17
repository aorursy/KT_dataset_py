# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import cross_validation, metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
train.head()

sns.countplot(x='Survived', hue='Sex', data = train)
sns.heatmap(train.isnull())
sns.countplot(x = 'Survived', hue = 'Sex', data = train)
sns.heatmap(train.isnull())
def fills_na(cols):
    Age = cols[0];
    pclass = cols[1];
    if pd.isnull(Age):
        if pclass==1:
            return 37
        if pclass==2:
            return 29
        else:
            return 24
    else:
        return Age   
        
        
train['Age']=train[['Age','Pclass']].apply(fills_na,axis=1)
sns.heatmap(train.isnull())
train=train.drop('Cabin',axis = 1)
train.head()
sex = pd.get_dummies(train['Sex'],drop_first=True)
embarked = pd.get_dummies(train['Embarked'],drop_first=True)
train = pd.concat([train,sex,embarked],axis=1)
train.drop(['Sex','Embarked','Name','Ticket','PassengerId'], axis=1, inplace=True)
x=train.drop('Survived',axis=1)
y=train['Survived']
from sklearn.linear_model import LogisticRegression
logis = LogisticRegression()
logis.fit(x,y)
test = pd.read_csv('../input/test.csv')
test['Age']=test[['Age','Pclass']].apply(fills_na,axis=1)
test.drop('Cabin',axis=1, inplace=True)
sex1=pd.get_dummies(test['Sex'],drop_first=True)
embarked1 = pd.get_dummies(test['Embarked'],drop_first=True)
test.drop(['PassengerId','Name','Sex','Ticket','Embarked'],axis=1,inplace=True)
test=pd.concat([test,sex1,embarked1],axis=1)
test.dropna()

test = pd.read_csv('../input/test.csv')
test['Age']=test[['Age','Pclass']].apply(fills_na,axis=1)
test.drop('Cabin',axis=1, inplace=True)
sex1=pd.get_dummies(test['Sex'],drop_first=True)
embarked1 = pd.get_dummies(test['Embarked'],drop_first=True)
test.drop(['PassengerId','Name','Sex','Ticket','Embarked'],axis=1,inplace=True)
test=pd.concat([test,sex1,embarked1],axis=1)
test=test.dropna()
predictions=logis.predict(test)
predictions