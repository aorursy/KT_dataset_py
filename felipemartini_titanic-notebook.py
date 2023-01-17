
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sb
import matplotlib.pyplot as pl

%matplotlib inline

train = pd.read_csv('/kaggle/input/titanic/train.csv')
test  = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
test.head()
train.shape
test.shape
train.info()
test.info()
train.isnull().sum()
test.isnull().sum()
def grafico(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar', stacked=True, figsize=(10,5))
grafico('Sex')
grafico('Pclass')
grafico('SibSp')
grafico('Parch')
grafico('Embarked')
train_test = [train, test]
for dataset in train_test:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train['Title'].value_counts()
test['Title'].value_counts()
title_map = {"Mr": 0,
            "Miss": 1,
            "Mrs": 2,
            "Master": 3,
            "Dr": 3,
            "Rev": 3,
            "Col": 3,
            "Major": 3,
            "Mlle": 3,
            "Ms": 3,
            "Don": 3,
            "Lady": 3,
            "Jonkheer": 3,
            "Countess": 3,
            "Mme": 3,
            "Sir": 3,
            "Capt": 3}
for dataset in train_test:
    dataset['Title'] = dataset['Title'].map(title_map)
train.head()
grafico('Title')
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)
test.head()
train.head()
sex_map = {"male": 0, "female": 1}
for dataset in train_test:
    dataset['Sex'] = dataset['Sex'].map(sex_map)
grafico('Sex')
train.head(100)
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)
facet = sb.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sb.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()


for dataset in train_test:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] =0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 62), 'Age'] = 4
grafico('Age')
Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1,Pclass2,Pclass3])
df.index = ['1st class', '2nd class', '3rd class']
df.plot(kind='bar', stacked=True, figsize=(10,5))
for dataset in train_test:
    dataset['Embarked'] =  dataset['Embarked'].fillna('S')
train.head()
emb_map = {"S": 0,
           "C": 1,
           "Q": 2}
for dataset in train_test:
    dataset['Embarked'] = dataset['Embarked'].map(emb_map)
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
for dataset in train_test:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] =0
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 100), 'Fare'] = 3
train.Cabin.value_counts()
for dataset in train_test:
    dataset['Cabin'] = dataset['Cabin'].str[:1]
Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1,Pclass2,Pclass3])
df.index = ['1st class', '2nd class', '3rd class']
df.plot(kind='bar', stacked=True, figsize=(10,5))
cab_map = {"A": 0,
           "B": 0.4,
           "C": 0.8,
           "D": 1.2,
           "E": 1.6,
           "F": 2.0,
           "G": 2.4,
           "T": 2.8}
for dataset in train_test:
    dataset['Cabin'] = dataset['Cabin'].map(cab_map)
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)