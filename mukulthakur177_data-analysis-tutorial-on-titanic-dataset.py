import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
train_set = pd.read_csv('../input/train-and-test-set/train.csv')
test_set = pd.read_csv('../input/train-and-test-set/test.csv')
train_set.head()
train_set.info()
test_set.head()
test_set.info()
sns.violinplot(x = 'Sex', y = 'Age', hue = 'Survived', data = train_set, split = True)
plt.show
sns.countplot('Sex', data = train_set, hue = 'Survived')
plt.show()
# Filling the empty colomns with the mean
for t_set in [train_set, test_set]:
    t_set['Age'].fillna(t_set['Age'].mean(), inplace=True)
    t_set['Fare'].fillna(t_set['Fare'].mean(), inplace=True)
    t_set['Embarked'].fillna(t_set['Embarked'].mode()[0], inplace=True)
train_set.info()
test_set.info()
for dataset in [train_set, test_set]:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# We can check the survival of people with different titles.
pd.crosstab(train_set['Title'], train_set['Survived'])
for dataset in [train_set, test_set]:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
# Checking the survivle of different titles
pd.crosstab(train_set['Title'], train_set['Survived'])
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5, "Rev": 6}
for dataset in [train_set, test_set]:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
for dataset in [train_set, test_set]:
    dataset['Sex'] = dataset['Sex'].map({"female": 1, "male": 2})
    dataset['Sex'] = dataset['Sex'].fillna(0)
for dataset in [train_set, test_set]:
    dataset['Embarked'] = dataset['Embarked'].map({"C": 1, "Q": 2, "S": 3})
    dataset['Embarked'] = dataset['Embarked'].fillna(0)
for cat in ['Cabin', 'Name', 'PassengerId', 'Ticket']:
    train_set = train_set.drop(cat, axis=1)
    test_set = test_set.drop(cat, axis=1)
train_set.head()
test_set.head()
for dataset in [train_set, test_set]:
    dataset['Age_grps'] = pd.cut(dataset['Age'], bins=[0,12,24,50,120], labels=[1, 2, 3, 4])
    dataset['Fare_bin'] = pd.cut(dataset['Fare'], bins=[-10,0,15,50,100,750], labels=[1, 2, 3, 4, 5])
for cat in ['Fare', 'Age']:
    train_set = train_set.drop(cat, axis=1)
    test_set = test_set.drop(cat, axis=1)
train_x = train_set.drop('Survived', axis=1)
train_y = train_set['Survived']
train_x.head()