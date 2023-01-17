# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression

import warnings

data_train = pd.read_csv("../input/train-with-cabin/trainCabin.csv")
data_test = pd.read_csv("../input/titanic/test.csv")
data_train
data_test
print(data_train.info())
print("--"*30)
print(data_test.info())
data_train.describe()
data_test.describe()
print(data_train.isnull().sum())
print('--'*20)
print(data_test.isnull().sum())
data_train[["Sex", "Survived"]].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.factorplot(x="Sex", y="Survived",  data=data_train,size=4, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("Survival Probability")
combine = [data_train, data_test]
sex_mapping = {'female': 1, 'male': 0}
for dataset in combine:    
    dataset['Sex'] = dataset['Sex'].map(sex_mapping).astype(int)
    dataset['Sex'] = dataset['Sex'].fillna(0)
data_train
data_train.loc[ (data_train.Cabin.notnull()), 'Cabin' ] = "Yes"
data_train.loc[ (data_train.Cabin.isnull()), 'Cabin' ] = "No"
data_test.loc[ (data_test.Cabin.notnull()), 'Cabin' ] = "Yes"
data_test.loc[ (data_test.Cabin.isnull()), 'Cabin' ] = "No"
data_train = data_train.drop(['Ticket'], axis=1)
data_test = data_test.drop(['Ticket'], axis=1)
combine = [data_train, data_test]
freq_port = data_train.Embarked.dropna().mode()[0]
freq_port

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].\
    map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
data_train
data_test
combine = [data_train, data_test]
for dataset in combine:
    dataset['Cabin'] = dataset['Cabin'].\
    map( {'No': 0, 'Yes': 1} ).astype(int)
    dataset['Cabin'] = dataset['Cabin'].fillna(0)
data_train
data_train['FareBand'] = pd.qcut(data_train['Fare'], 4)
data_train[['FareBand', 'Survived']].groupby(['FareBand'],as_index=False).mean().sort_values(by='FareBand', ascending=True)
data_test['Fare'].fillna(data_test['Fare'].dropna().median(), inplace=True)
combine = [data_train,data_test]

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
from sklearn.ensemble import RandomForestRegressor
def set_missing_ages(df):
    
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values

    y = known_age[:, 0]

    X = known_age[:, 1:]

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    
    predictedAges = rfr.predict(unknown_age[:, 1::])
    
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    
    return df, rfr

data_train, rfr = set_missing_ages(data_train)
age_dt = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

known_age = age_dt[age_dt.Age.notnull()].values
unknown_age = age_dt[age_dt.Age.isnull()].values

predictedAges = rfr.predict(unknown_age[:, 1::])

data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges 
data_train['AgeBand'] = pd.qcut(data_train['Age'], 5)
data_train[['AgeBand', 'Survived']].groupby(['AgeBand'],as_index=False).mean().sort_values(by='AgeBand', ascending=True)
combine = [data_train, data_test]
for dataset in combine:
    dataset.loc[(dataset['Age'] >= 0.419) & (dataset['Age'] <= 19.831), 'Age'] = 0
    dataset.loc[(dataset['Age'] > 19.831) & (dataset['Age'] <= 27.0), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 27.0) & (dataset['Age'] <= 30.0), 'Age']   = 2
    dataset.loc[(dataset['Age'] > 30.0) & (dataset['Age'] <= 39.0), 'Age'] = 3
    dataset.loc[ (dataset['Age'] > 39.0), 'Age'] = 4
    dataset['Age'] = dataset['Age'].astype(int)
data_train = data_train.drop(['FareBand'], axis=1)
data_train = data_train.drop(['AgeBand'], axis=1)
data_train
data_test
combine = [data_train, data_test]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.',expand=False)

pd.crosstab(data_train['Title'], data_train['Sex'])
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

data_train[['Title', 'Survived']].groupby(['Title'], as_index=False,).mean().sort_values(by='Title', ascending=True)
combine = [data_train, data_test]
title_mapping = {'Mrs': 4, 'Miss': 3, 'Master': 2, 'Rare': 1, 'Mr': 0}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping).astype(int)
data_train = data_train.drop(['Name'], axis=1)
data_test = data_test.drop(['Name'], axis=1)
data_test
data_train
