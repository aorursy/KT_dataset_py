# Import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
PATH = '../input/'
train_data = pd.read_csv(PATH + 'train.csv')

test_data = pd.read_csv(PATH + 'test.csv')

gender_submission = pd.read_csv(PATH + 'gender_submission.csv')
train_data.head()
test_data.head()
train_data.describe()
train_data.columns
train_data.dtypes
column_names = train_data.columns

for column in column_names:

    print(column + ' - ' + str(train_data[column].isnull().sum()))
train_data.Survived.value_counts()
plt = train_data.Survived.value_counts().plot('bar')

plt.set_xlabel('Survived or not')

plt.set_ylabel('Passenger Count')
plt = train_data.Pclass.value_counts().sort_index().plot('bar', title='')

plt.set_xlabel('Pclass')

plt.set_ylabel('Survival Probability')
train_data[['Pclass', 'Survived']].groupby('Pclass').count()
train_data[['Pclass', 'Survived']].groupby('Pclass').sum()
plt = train_data[['Pclass', 'Survived']].groupby('Pclass').mean().Survived.plot('bar')

plt.set_xlabel('Pclass')

plt.set_ylabel('Survival Probability')
plt = train_data.Sex.value_counts().sort_index().plot('bar')

plt.set_xlabel('Sex')

plt.set_ylabel('Passenger count')
plt = train_data[['Sex', 'Survived']].groupby('Sex').mean().Survived.plot('bar')

plt.set_xlabel('Sex')

plt.set_ylabel('Survival Probability')
plt = train_data.Embarked.value_counts().sort_index().plot('bar')

plt.set_xlabel('Embarked')

plt.set_ylabel('Passenger count')
plt = train_data[['Embarked', 'Survived']].groupby('Embarked').mean().Survived.plot('bar')

plt.set_xlabel('Embarked')

plt.set_ylabel('Survival Probability')
plt = train_data.SibSp.value_counts().sort_index().plot('bar')

plt.set_xlabel('SibSp')

plt.set_ylabel('Passenger count')
plt = train_data[['SibSp', 'Survived']].groupby('SibSp').mean().Survived.plot('bar')

plt.set_xlabel('SibSp')

plt.set_ylabel('Survival Probability')
plt = train_data.Parch.value_counts().sort_index().plot('bar')

plt.set_xlabel('Parch')

plt.set_ylabel('Passenger count')
plt = train_data[['Parch', 'Survived']].groupby('Parch').mean().Survived.plot('bar')

plt.set_xlabel('Parch')

plt.set_ylabel('Survival Probability')
sns.factorplot('Pclass', col = 'Embarked', data = train_data, kind = 'count')
sns.factorplot('Sex', col = 'Pclass', data = train_data, kind = 'count')
sns.factorplot('Sex', col = 'Embarked', data = train_data, kind = 'count')
train_data.head()
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
train_data.head()
train_data = train_data.drop(columns=['Ticket', 'PassengerId', 'Cabin'])
train_data.head()
train_data['Sex'] = train_data['Sex'].map({'male':0, 'female':1})

train_data['Embarked'] = train_data['Embarked'].map({'C':0, 'Q':1, 'S':2})
train_data.head()
train_data['Title'] = train_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

train_data = train_data.drop(columns='Name')
train_data.Title.value_counts().plot('bar')
train_data['Title'] = train_data['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don'], 'Others')

train_data['Title'] = train_data['Title'].replace('Ms', 'Miss')

train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs')

train_data['Title'] = train_data['Title'].replace('Mlle', 'Miss')
plt = train_data.Title.value_counts().sort_index().plot('bar')

plt.set_xlabel('Title')

plt.set_ylabel('Passenger count')
plt = train_data[['Title', 'Survived']].groupby('Title').mean().Survived.plot('bar')

plt.set_xlabel('Title')

plt.set_ylabel('Survival Probability')
train_data['Title'] = train_data['Title'].map({'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Others':4})
train_data.head()
corr_matrix = train_data.corr()
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 8))

sns.heatmap(data = corr_matrix,cmap='BrBG', annot=True, linewidths=0.2)
train_data.isnull().sum()
train_data['Embarked'].isnull().sum()
train_data['Embarked'] = train_data['Embarked'].fillna(2)

train_data.head()
corr_matrix = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].corr()
plt.figure(figsize=(7, 6))

sns.heatmap(data = corr_matrix,cmap='BrBG', annot=True, linewidths=0.2)
NaN_indexes = train_data['Age'][train_data['Age'].isnull()].index
for i in NaN_indexes:

    pred_age = train_data['Age'][((train_data.SibSp == train_data.iloc[i]["SibSp"]) & (train_data.Parch == train_data.iloc[i]["Parch"]) & (train_data.Pclass == train_data.iloc[i]["Pclass"]))].median()

    if not np.isnan(pred_age):

        train_data['Age'].iloc[i] = pred_age

    else:

        train_data['Age'].iloc[i] = train_data['Age'].median()
train_data.isnull().sum()
train_data.head()
test_data = pd.read_csv(PATH + 'test.csv')
test_data.isnull().sum()
test_data = test_data.drop(columns=['Ticket', 'PassengerId', 'Cabin'])
test_data.head()
test_data['Sex'] = test_data['Sex'].map({'male':0, 'female':1})

test_data['Embarked'] = test_data['Embarked'].map({'C':0, 'Q':1, 'S':2})
test_data.head()
test_data['Title'] = test_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

test_data = test_data.drop(columns='Name')



test_data['Title'] = test_data['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don'], 'Others')

test_data['Title'] = test_data['Title'].replace('Ms', 'Miss')

test_data['Title'] = test_data['Title'].replace('Mme', 'Mrs')

test_data['Title'] = test_data['Title'].replace('Mlle', 'Miss')



test_data['Title'] = test_data['Title'].map({'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Others':4})
test_data.head()
test_data.isnull().sum()
NaN_indexes = test_data['Age'][test_data['Age'].isnull()].index



for i in NaN_indexes:

    pred_age = train_data['Age'][((train_data.SibSp == test_data.iloc[i]["SibSp"]) & (train_data.Parch == test_data.iloc[i]["Parch"]) & (test_data.Pclass == train_data.iloc[i]["Pclass"]))].median()

    if not np.isnan(pred_age):

        test_data['Age'].iloc[i] = pred_age

    else:

        test_data['Age'].iloc[i] = train_data['Age'].median()
title_mode = train_data.Title.mode()[0]

test_data.Title = test_data.Title.fillna(title_mode)
fare_mean = train_data.Fare.mean()

test_data.Fare = test_data.Fare.fillna(fare_mean)
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
test_data.head()
train_data.head()
from sklearn.utils import shuffle

train_data = shuffle(train_data)
# training_data, valid_data = train_test_split(train_data, test_size=0.2)
X_train = train_data.drop(columns='Survived')

y_train = train_data.Survived

y_train = pd.DataFrame({'Survived':y_train.values})
# X_valid = valid_data.drop(columns='Survived')

# y_valid = valid_data.Survived
X_test = test_data
X_train.head()
y_train.head()
X_train.shape
y_train.shape
X_test.head()
X_train.to_csv('X_train.csv', index=False)

y_train.to_csv('y_train.csv', index=False)



# X_valid.to_csv('X_valid.csv', index=False)

# y_valid.to_csv('y_valid.csv', index=False)



X_test.to_csv('X_test.csv', index=False)