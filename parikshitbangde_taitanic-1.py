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
# importing libraies 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train.head()
test.head()
# checking the shape

print ('train dataset shape ',train.shape,'\n','test dataset shape',test.shape)
train['Cabin'].mode()
train['Age'] = train['Age'].fillna(train['Age'].mean())

train['Embarked'] = train['Embarked'].fillna('S')

train['Cabin'] = train['Cabin'].fillna(train['Cabin'].mode())
# out liers 

sns.distplot(train['Age'])

train['Age'].value_counts().sort_values(ascending=False)

print ('min age ',train['Age'].min(), '\n','max age ',train['Age'].max()) 
# checking for NAN values 

print ('Nan value present in train set ', train.isna().sum())
test['Age'] = test['Age'].fillna(test['Age'].mean())

test['Fare'] = test['Fare'].fillna(test['Fare'].mean())



test['Cabin'] = test['Cabin'].fillna(test['Cabin'].mode())
sns.countplot(test['Age'])

test['Age'].value_counts().sort_values(ascending=False)
print ('min age ',test['Age'].min(), '\n','max age ',test['Age'].max()) 
print ('Nan value present in test set ', test.isna().sum())
train['Age'].unique()
test['Age'].unique()
train['Embarked'].value_counts()
test['Cabin'] = test['Cabin'].astype('str')

train['Cabin'] = train['Cabin'].astype('str')
set(test['Cabin']) - set(train['Cabin'])
X = train.drop(columns=['Survived'],axis=1)
y = train['Survived']
data = pd.concat(objs=[X,test])
data.shape
data['Cabin'].value_counts()
data = data.drop(columns=['Name','Embarked'])

dummies = pd.get_dummies(data)

dummies.shape
"""

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer = imputer.fit(dummies)

imputed_data= imputer.transform(dummies)

imputed_data = pd.DataFrame(imputed_data)

"""

"""

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

#data['Embarked']= le.fit_transform(data['Embarked'])

data['Sex']= le.fit_transform(data['Sex'])

data['Ticket']= le.fit_transform(data['Ticket'])

data['Cabin']= le.fit_transform(data['Cabin'])

"""
training  = dummies.iloc[:len(X)]

test = dummies.iloc[len(X):]
print (training.shape,test.shape)
print ('Nan value present in data set ', data.isna().sum())
from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=1000,n_jobs=-1);



bclf = AdaBoostClassifier(base_estimator=clf,n_estimators=clf.n_estimators)

training.info()
bclf.fit(training,y)
bclf.score(training,y)
pred = bclf.predict(test)

test.head()
my_submission = pd.DataFrame({'PassengerId': test.PassengerId.astype('int64'), 'Survived':pred})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)
pred