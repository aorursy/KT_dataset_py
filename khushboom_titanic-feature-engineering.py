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
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
test.head()
train.shape

test.shape
train.describe()
train.info()
#we can see that Age, cabin and Embarked predictors have missing values

train.isnull().sum()
# exploratory data analysis
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
survived = train[train['Survived']==1]['Sex'].value_counts()
dead = train[train['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived,dead])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True)
#Women more likely survivied than Men
survived = train[train['Survived']==1]['Pclass'].value_counts()
dead = train[train['Survived']==0]['Pclass'].value_counts()
df = pd.DataFrame([survived,dead])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True)
#1st class more likely to survive than other classes
#3rd class more likely to be dead than other classes
survived = train[train['Survived']==1]['SibSp'].value_counts()
dead = train[train['Survived']==0]['SibSp'].value_counts()
df = pd.DataFrame([survived,dead])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True)
#a person having no sibling or spouse on board are more likely to be dead
train_data = pd.get_dummies(train, columns = ['Sex'], drop_first=True)
train_data.head()
# getting dummies where in column Sex_male 1 means male
median = train_data['Age'].median()
train_data['Age'].fillna(median, inplace =True)
test_data = pd.get_dummies(test, columns = ['Sex'], drop_first=True)
median_test = test_data['Age'].median()
test_data['Age'].fillna(median_test, inplace =True)
train_data.isnull().sum()
Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))
train_data['Embarked'].fillna('S', inplace =True)
test_data['Embarked'].fillna('S', inplace = True)
test_data.isnull().sum()
train_data.isnull().sum()
train_data.dropna(axis='columns')
test_data.drop('Cabin', axis=1, inplace=True)
train_data.dropna()
train_data.drop('Cabin', axis=1, inplace=True)
test_data.dropna()

train_data["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test_data["FamilySize"] = test["SibSp"] + test["Parch"] + 1
features = ['Ticket', 'SibSp', 'Parch']
train_data = train_data.drop(features, axis=1)
test_data = test_data.drop(features, axis=1)
train_data = train_data.drop(['PassengerId'], axis=1)
train_data_final = train_data.drop('Survived', axis=1)
Predict = train_data['Survived']

Predict.shape
train_data_final.head()
embarked = {'S': 0, 'C': 1, 'Q': 2}
for data in train_data_final:
    data['Embarked'] = data['Embarked'].map(embarked)
for dataset in train_data_final:
    if dataset['Age'] <= 16:
        dataset['Age'] = 0
    elif dataset['Age'] > 16 & dataset['Age'] <= 26:
        dataset['Age'] = 1
    elif dataset['Age'] > 26 & dataset['Age'] <= 40:
        dataset['Age'] = 2
    elif dataset['Age'] > 40 & dataset['Age'] <= 60:
        dataset['Age'] = 3
    elif dataset['Age'] > 60:
        dataset['Age'] = 4
