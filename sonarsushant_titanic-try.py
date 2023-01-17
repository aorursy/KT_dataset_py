import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train=pd.read_csv('../input/train.csv')
train.head()
sns.heatmap(train.isnull(),cbar=False,yticklabels=False)
def impute_age(cols):

    Age=cols[0]

    Pclass=cols[1]

    if pd.isnull(Age):

        if Pclass==1:

            return 37

        elif Pclass==2:

            return 29

        else:

            return 24

    else:

        return Age
train['Age']=train[['Age','Pclass']].apply(impute_age,axis=1)

train.drop('Cabin',axis=1,inplace=True)
train.dropna(inplace=True)
sns.heatmap(train.isnull(),cbar=False,yticklabels=False)
sns.countplot(x='Survived',data=train,hue='Sex')

sns.countplot(x='Survived',data=train,hue='Pclass')

sns.distplot(train['Age'],kde=False,bins=30)
sns.countplot(x='SibSp',data=train)
sns.countplot(x='Embarked',data=train)
sex=pd.get_dummies(train['Sex'],drop_first=True)

embark=pd.get_dummies(train['Embarked'],drop_first=True)

train=pd.concat([train,sex,embark],axis=1)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
X_train=train.drop('Survived',axis=1)

y_train=train['Survived']
X_train.drop('PassengerId',axis=1,inplace=True)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
test_data=pd.read_csv('../input/test.csv')
test_data['Age']=test_data[['Age','Pclass']].apply(impute_age,axis=1)
test_data.drop('Cabin',axis=1,inplace=True)
test_data.dropna(inplace=True)
sex=pd.get_dummies(test_data['Sex'],drop_first=True)

embark=pd.get_dummies(test_data['Embarked'],drop_first=True)

test_data=pd.concat([test_data,sex,embark],axis=1)
test_data.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
X_test=test_data.drop('PassengerId',axis=1)
predictions=logmodel.predict(X_test)
test_data['Survived']=predictions