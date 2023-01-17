# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

full_data = [train, test]
train.info()
print(train[['Sex', 'Survived']].groupby(['Sex']).describe())
print(train[['Pclass', 'Survived']].groupby(['Pclass']).describe())
print(train[['Fare', 'Survived']].groupby(['Fare']).describe().head(20))
df = train.copy()

df['RoundFare'] = train['Fare'].round(-1).astype(int)

print(df[['RoundFare', 'Survived']].groupby(['RoundFare']).describe())
for dataset in full_data:

    boundary_value = 50

    dataset['CategoryFare'] = dataset['Fare'].apply(lambda x: 0 if x <= boundary_value else 1)

print(train[['CategoryFare', 'Survived']].groupby(['CategoryFare']).describe())
print(train[['Embarked', 'Survived']].groupby(['Embarked']).describe())
for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

print(train[['Embarked', 'Survived']].groupby(['Embarked']).describe())
for dataset in full_data:

    dataset['FamirySize'] = dataset['SibSp'] + dataset['Parch'] + 1

print(train[['Sex', 'FamirySize', 'Survived']].groupby(['Sex', 'FamirySize']).describe())
for dataset in full_data:

    boundary_value = 4

    dataset['IsBigFamiry'] = dataset['FamirySize'].apply(lambda x: 0 if x <= boundary_value else 1)

print(train[['Sex', 'IsBigFamiry', 'Survived']].groupby(['Sex', 'IsBigFamiry']).describe())
print(train[['Age', 'Survived']].groupby(['Age']).describe())
df = train.copy()

df['AgeGroups'] = pd.cut(df['Age'], np.arange(0, 201, 10))

print(df[['Sex', 'Pclass', 'AgeGroups', 'Survived']].groupby(['Sex', 'Pclass', 'AgeGroups']).describe())
alone = df.loc[(df['FamirySize'] == 1)]

print(alone[['Pclass', 'Sex', 'Age']].groupby(['Pclass', 'Sex']).describe())
famiry = df.loc[df['FamirySize'] > 1]

print(famiry[['FamirySize','Pclass', 'Sex', 'Age']].groupby(['FamirySize', 'Pclass', 'Sex']).describe())
df['RoundFamirySize'] = df['FamirySize'].apply(lambda x: x if x <= 3 else 4)

print(df[['RoundFamirySize','Pclass', 'Sex', 'Age']].groupby(['RoundFamirySize', 'Pclass', 'Sex']).describe())
for dataset in full_data:

    dataset['RoundFamirySize'] = dataset['FamirySize'].apply(lambda x: x if x <= 3 else 4)

    # RoundFamirySize = 1

    dataset.loc[(dataset['Age'].isnull()) 

                & (dataset['RoundFamirySize'] == 1) 

                & (dataset['Pclass'] == 1) 

                & (dataset['Sex'] == 'male')

                , 'Age'] = 44

     

    dataset.loc[(dataset['Age'].isnull()) 

                & (dataset['RoundFamirySize'] == 1) 

                & (dataset['Pclass'] == 1) 

                & (dataset['Sex'] == 'female')

                , 'Age'] = 34



    dataset.loc[(dataset['Age'].isnull()) 

                & (dataset['RoundFamirySize'] == 1) 

                & (dataset['Pclass'] == 2) 

                , 'Age'] = 33

    

    dataset.loc[(dataset['Age'].isnull()) 

                & (dataset['RoundFamirySize'] == 1) 

                & (dataset['Pclass'] == 3) 

                & (dataset['Sex'] == 'male')

                , 'Age'] = 29

    

    dataset.loc[(dataset['Age'].isnull()) 

                & (dataset['RoundFamirySize'] == 1) 

                & (dataset['Pclass'] == 3) 

                & (dataset['Sex'] == 'female')

                , 'Age'] = 23

    

    # RoundFamirySize = 2

    dataset.loc[(dataset['Age'].isnull()) 

                & (dataset['RoundFamirySize'] == 2) 

                & (dataset['Pclass'] == 1) 

                & (dataset['Sex'] == 'male')

                , 'Age'] = 37

    

    dataset.loc[(dataset['Age'].isnull()) 

                & (dataset['RoundFamirySize'] == 2) 

                & (dataset['Pclass'] == 1) 

                & (dataset['Sex'] == 'female')

                , 'Age'] = 36

    

    dataset.loc[(dataset['Age'].isnull()) 

                & (dataset['RoundFamirySize'] == 2) 

                & (dataset['Pclass'] == 2) 

                & (dataset['Sex'] == 'male')

                , 'Age'] = 33

    

    dataset.loc[(dataset['Age'].isnull()) 

                & (dataset['RoundFamirySize'] == 2) 

                & (dataset['Pclass'] == 2) 

                & (dataset['Sex'] == 'female')

                , 'Age'] = 28

    

    dataset.loc[(dataset['Age'].isnull()) 

                & (dataset['RoundFamirySize'] == 2) 

                & (dataset['Pclass'] == 3) 

                & (dataset['Sex'] == 'male')

                , 'Age'] = 24

    

    dataset.loc[(dataset['Age'].isnull()) 

                & (dataset['RoundFamirySize'] == 2) 

                & (dataset['Pclass'] == 3) 

                & (dataset['Sex'] == 'female')

                , 'Age'] = 22

    

    # RoundFamirySize = 3

    dataset.loc[(dataset['Age'].isnull()) 

                & (dataset['RoundFamirySize'] == 3) 

                & (dataset['Pclass'] == 1) 

                & (dataset['Sex'] == 'male')

                , 'Age'] = 42

    

    dataset.loc[(dataset['Age'].isnull()) 

                & (dataset['RoundFamirySize'] == 3) 

                & (dataset['Pclass'] == 1) 

                & (dataset['Sex'] == 'female')

                , 'Age'] = 36

    

    dataset.loc[(dataset['Age'].isnull()) 

                & (dataset['RoundFamirySize'] == 3) 

                & (dataset['Pclass'] == 2) 

                & (dataset['Sex'] == 'male')

                , 'Age'] = 18

    

    dataset.loc[(dataset['Age'].isnull()) 

                & (dataset['RoundFamirySize'] == 3) 

                & (dataset['Pclass'] == 2) 

                & (dataset['Sex'] == 'female')

                , 'Age'] = 22

    

    dataset.loc[(dataset['Age'].isnull()) 

                & (dataset['RoundFamirySize'] == 3) 

                & (dataset['Pclass'] == 3) 

                & (dataset['Sex'] == 'male')

                , 'Age'] = 23

    

    dataset.loc[(dataset['Age'].isnull()) 

                & (dataset['RoundFamirySize'] == 3) 

                & (dataset['Pclass'] == 3) 

                & (dataset['Sex'] == 'female')

                , 'Age'] = 20

    

    # RoundFamirySize = 4

    dataset.loc[(dataset['Age'].isnull()) 

                & (dataset['RoundFamirySize'] == 4) 

                & (dataset['Pclass'] == 1) 

                & (dataset['Sex'] == 'male')

                , 'Age'] = 26

    

    dataset.loc[(dataset['Age'].isnull()) 

                & (dataset['RoundFamirySize'] == 4) 

                & (dataset['Pclass'] == 1) 

                & (dataset['Sex'] == 'female')

                , 'Age'] = 20

    

    dataset.loc[(dataset['Age'].isnull()) 

                & (dataset['RoundFamirySize'] == 4) 

                & (dataset['Pclass'] == 2) 

                & (dataset['Sex'] == 'male')

                , 'Age'] = 21

    

    dataset.loc[(dataset['Age'].isnull()) 

                & (dataset['RoundFamirySize'] == 4) 

                & (dataset['Pclass'] == 2) 

                & (dataset['Sex'] == 'female')

                , 'Age'] = 24

    

    

    dataset.loc[(dataset['Age'].isnull()) 

                & (dataset['RoundFamirySize'] == 4) 

                & (dataset['Pclass'] == 3) 

                & (dataset['Sex'] == 'male')

                , 'Age'] = 10

    

    dataset.loc[(dataset['Age'].isnull()) 

                & (dataset['RoundFamirySize'] == 4) 

                & (dataset['Pclass'] == 3) 

                & (dataset['Sex'] == 'female')

                , 'Age'] = 19
for dataset in full_data:

    dataset['CategoryAge'] = None

    dataset.loc[dataset['Age'] <= 10, 'CategoryAge'] = 0

    dataset.loc[(10 < dataset['Age']) & (dataset['Age'] <= 40), 'CategoryAge'] = 0

    dataset.loc[40 < dataset['Age'], 'CategoryAge'] = 0
print(train[['Cabin', 'Survived']].groupby(['Cabin']).describe().head(10))
print(train[['Ticket', 'Survived']].groupby(['Ticket']).describe().head(10))
print(train[['Name', 'Survived']].groupby(['Name']).describe().head(5))
for dataset in full_data:

    dataset['Sex'] = dataset['Sex'].map({'male':0, 'female':1}).astype(int)

    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

    dataset['Pclass'] = dataset['Pclass'].map({1:0, 2:1, 3:2}).astype(int)
drop_col = ['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 

            'FamirySize', 'RoundFamirySize']

    

train = train.drop(drop_col, axis=1)

test = test.drop(drop_col, axis=1)
x_train = train.drop("Survived", axis=1).copy()

y_train = train["Survived"].copy()

x_test  = test.copy()

print(x_train.shape, y_train.shape, x_test.shape)
train.columns
logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

acc_log = round(logreg.score(x_train, y_train) * 100, 2)

acc_log
# Support Vector Machines

svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

acc_svc = round(svc.score(x_train, y_train) * 100, 2)

acc_svc
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(x_train, y_train)

y_pred = random_forest.predict(x_test)

random_forest.score(x_train, y_train)

acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)

acc_random_forest