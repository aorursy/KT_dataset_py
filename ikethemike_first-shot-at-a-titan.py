# data analysis packages

import pandas as pd

import numpy as np

import random as rnd



# get the data using pandas



train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

train_df.head()
print(train_df.columns.values)
train_df.info()

print('---------')

test_df.info()

train_df.describe(include=['O'])

train_df['Embarked'] = train_df['Embarked'].fillna('S')

test_df['Embarked'] = test_df['Embarked'].fillna('S')



train_df['Fare'].fillna(train_df['Fare'].dropna().median(), inplace=True)

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)



train_df =  train_df.drop(['Ticket','Cabin','Name','PassengerId'], axis=1)

test_df =  test_df.drop(['Ticket','Cabin','Name','PassengerId'], axis=1)

train_df['Age'].fillna(train_df['Age'].dropna().median(), inplace= True)

test_df['Age'].fillna(test_df['Age'].dropna().median(), inplace= True)
train_df.describe()
train_df['AgeGroups'] = pd.cut(train_df['Age'],5)

train_df[['AgeGroups', 'Survived']].groupby(['AgeGroups'], as_index=False).mean().sort_values(by='AgeGroups', ascending=True)
for dataset in [train_df,test_df]:

    dataset.loc[ dataset["Age"] <= 16,'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()
train_df = train_df.drop(['AgeGroups'],axis=1)

train_df.head()
train_df['Sex'] = train_df['Sex'].map({'male':1,'female':0}).astype(int)

test_df['Sex'] = test_df['Sex'].map({'male':1,'female':0}).astype(int)

train_df.head()
for dataset in [train_df,test_df]:

    dataset['Has_relatives'] = 0

    dataset.loc[dataset['SibSp']+ dataset['Parch'] +1 > 1,'Has_relatives'] = 1

    dataset['Has_relatives'] = dataset['Has_relatives'].astype(int)

train_df.head()
train_df = train_df.drop(['SibSp','Parch'],axis=1)

test_df = test_df.drop(['SibSp','Parch'],axis=1)



train_df.head()
train_df['Embarked'] = train_df['Embarked'].map({'S':1,'C':0,'Q':2}).astype(int)

test_df['Embarked']= test_df['Embarked'].map({'S':1,'C':0,'Q':2}).astype(int)



train_df.head()
train_df['FareGroups'] = pd.qcut(train_df['Fare'],4)

train_df[['FareGroups','Survived']].groupby(['FareGroups'],as_index=False).mean().sort_values(by='FareGroups',ascending=True)

for dataset in [train_df,test_df]:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df = train_df.drop(['FareGroups'], axis=1)

    

train_df.head(10)