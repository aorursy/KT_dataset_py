# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Importing necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
#Reading the train and test data

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
#Checking data

train_df.head()
test_df.head()
#Check info

train_df.info()

print('-----------------------------------------------------------------')

test_df.info()
#Remove the target feature i.e. Survived

y1 = train_df['Survived'].copy()

train_df.drop(columns=['Survived'],inplace=True)
train_df.describe()
train_df.isnull().sum()
test_df.isnull().sum()
# We have missing values in Age, Cabin and Embarked

# Cabin has a lot of missing values so we will drop the column

train_df.drop(columns=['Cabin'],inplace=True)

test_df.drop(columns=['Cabin'],inplace=True)

# Age is a numerical data, so we will use median to fill the missing values

train_df['Age'].fillna(train_df['Age'].median(),inplace=True)

test_df['Age'].fillna(test_df['Age'].median(),inplace=True)



# Fare in test data

test_df['Fare'].fillna(test_df['Fare'].median(),inplace=True)
# Embarked in train data

# Find category and it's counts

train_df['Embarked'].value_counts().plot(kind='bar',figsize=(6,4), title='Embarked')
# We will replace S which has the highest frequency in the place of missing values

train_df['Embarked'].fillna('S',inplace=True)
# PassengerId, Ticket are not of any use

train_df.drop(columns=['PassengerId','Ticket'],inplace=True)

test_df.drop(columns=['PassengerId','Ticket'],inplace=True)

# Feature Enginnering

train_df['Title'] = train_df['Name'].apply(lambda x: x[x.find(',')+2:x.find('.')])

test_df['Title'] = test_df['Name'].apply(lambda x: x[x.find(',')+2:x.find('.')])
train_df['Title'] = train_df['Title'].replace(['Mme','Ms'],'Mrs')

train_df['Title'] = train_df['Title'].replace(['Mlle','Lady'],'Miss')

train_df['Title'] = train_df['Title'].replace(['the Countess',

                                               'Capt', 'Col','Don', 

                                               'Dr', 'Major', 'Rev', 

                                               'Sir', 'Jonkheer', 'Dona'], 'Others')



test_df['Title'] = test_df['Title'].replace(['Mme','Ms'],'Mrs')

test_df['Title'] = test_df['Title'].replace(['Mlle','Lady'],'Miss')

test_df['Title'] = test_df['Title'].replace(['the Countess',

                                               'Capt', 'Col','Don', 

                                               'Dr', 'Major', 'Rev', 

                                               'Sir', 'Jonkheer', 'Dona'], 'Others')
train_df['Title'].value_counts().plot(kind='bar', figsize=(6,4), title='Title')
test_df['Title'].value_counts().plot(kind='bar', figsize=(6,4), title='Title')
# Drop Name Column

train_df.drop(columns=['Name'],inplace=True)

test_df.drop(columns=['Name'],inplace=True)
train_df = pd.get_dummies(columns=['Title','Embarked','Sex'],data=train_df)

test_df = pd.get_dummies(columns=['Title','Embarked','Sex'],data=test_df)
#Training data

x = train_df.iloc[:,:].values

y = y1.values



#Test data

test = test_df.iloc[:,:].values
from sklearn.model_selection import train_test_split

xtr,xvl,ytr,yvl = train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)

rf.fit(xtr,ytr)

rf.score(xtr,ytr)
y_pred = rf.predict(xvl)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred,yvl)

cm
y_pred2 = rf.predict(test)
submission = pd.read_csv('../input/gender_submission.csv')

df = pd.DataFrame({'Survived':y_pred2})

submission.update(df)
from lightgbm import LGBMClassifier
lgb = LGBMClassifier(objective='binary',random_state=0)

lgb.fit(x,y)

lgb.score(x,y)
y_pred2 = lgb.predict(test)

submission = pd.read_csv('../input/gender_submission.csv')

df = pd.DataFrame({'Survived':y_pred2})

submission.update(df)