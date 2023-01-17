import pandas as pd

import numpy as np

%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import norm

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train=pd.read_csv('/kaggle/input/titanic/train.csv', index_col='PassengerId')

test=pd.read_csv('/kaggle/input/titanic/test.csv', index_col='PassengerId')

submission=pd.read_csv('/kaggle/input/titanic/gender_submission.csv', index_col='PassengerId')



print(train.shape, test.shape, submission.shape)
train.isnull().sum()
test.isnull().sum()
train=train.drop(columns='Cabin')

test=test.drop(columns='Cabin')
sns.countplot(data=train, x='Sex', hue='Survived')
train.loc[train['Sex']=='male', 'Sex']=0

train.loc[train['Sex']=='female','Sex']=1

test.loc[test['Sex']=='male','Sex']=0

test.loc[test['Sex']=='female','Sex']=1
sns.countplot(data=train, x='Pclass', hue='Survived')
train['Pclass_3']=(train['Pclass']==3)

train['Pclass_2']=(train['Pclass']==2)

train['Pclass_1']=(train['Pclass']==1)



test['Pclass_3']=(test['Pclass']==3)

test['Pclass_2']=(test['Pclass']==2)

test['Pclass_1']=(test['Pclass']==1)
train=train.drop(columns='Pclass')

test=test.drop(columns='Pclass')
sns.lmplot(data=train, x='Age', y='Fare', fit_reg=False, hue='Survived')
LowFare=train[train['Fare']<80]

sns.lmplot(data=LowFare, x='Age', y='Fare', hue='Survived')
test.loc[test['Fare'].isnull(),'Fare']=0
train=train.drop(columns='Age')

test=test.drop(columns='Age')
train['FamilySize']=train['SibSp']+train['Parch']+1

test['FamilySize']=test['SibSp']+test['Parch']+1



figure, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

figure.set_size_inches(18,6)

sns.countplot(data=train, x='SibSp', hue='Survived', ax=ax1)

sns.countplot(data=train, x='Parch', hue='Survived', ax=ax2)

sns.countplot(data=train, x='FamilySize',hue='Survived', ax=ax3)
train['Single']=train['FamilySize']==1

train['Nuclear']=(2<=train['FamilySize']) & (train['FamilySize']<=4)

train['Big']=train['FamilySize']>=5



test['Single']=test['FamilySize']==1

test['Nuclear']=(2<=test['FamilySize']) & (test['FamilySize']<=4)

test['Big']=test['FamilySize']>=5



figure, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

figure.set_size_inches(18,6)

sns.countplot(data=train, x='Single', hue='Survived', ax=ax1)

sns.countplot(data=train, x='Nuclear', hue='Survived', ax=ax2)

sns.countplot(data=train, x='Big',hue='Survived', ax=ax3) 
train=train.drop(columns=['Single','Big','SibSp','Parch','FamilySize'])

test=test.drop(columns=['Single','Big','SibSp','Parch','FamilySize'])
sns.countplot(data=train, x='Embarked', hue='Survived')
train['EmbarkedC']=train['Embarked']=='C'

train['EmbarkedS']=train['Embarked']=='S'

train['EmbarkedQ']=train['Embarked']=='Q'

test['EmbarkedC']=test['Embarked']=='C'

test['EmbarkedS']=test['Embarked']=='S'

test['EmbarkedQ']=test['Embarked']=='Q'
train=train.drop(columns='Embarked')

test=test.drop(columns='Embarked')
train['Name']=train['Name'].str.split(', ').str[1].str.split('. ').str[0]

test['Name']=test['Name'].str.split(', ').str[1].str.split('. ').str[0]
sns.countplot(data=train, x='Name', hue='Survived')
train['Master']=(train['Name']=='Master')

test['Master']=(test['Name']=='Master')



train=train.drop(columns='Name')

test=test.drop(columns='Name')
train=train.drop(columns='Ticket')

test=test.drop(columns='Ticket')
from sklearn.tree import DecisionTreeClassifier
Ytrain=train['Survived']

feature_names=list(test)

Xtrain=train[feature_names]

Xtest=test[feature_names]



print(Xtrain.shape, Ytrain.shape, Xtest.shape)

Xtrain.head()
model=DecisionTreeClassifier(max_depth=8, random_state=18)
model.fit(Xtrain, Ytrain)

predictions=model.predict(Xtest)

submission['Survived']=predictions

submission.to_csv('Result.csv')

submission.head()