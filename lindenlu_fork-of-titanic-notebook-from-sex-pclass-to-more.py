# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



#load training and testing dataframe from input

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df]
#Sex

#convert categorical features to numerical

for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map({'male':0,'female':1}).astype(int)
#Title

#extract title from name

train_df['Title'] = train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

test_df['Title'] = test_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

#group titles

train_df['Title'] = train_df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

test_df['Title'] = test_df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

train_df['Title'] = train_df['Title'].replace('Mlle', 'Miss')

train_df['Title'] = train_df['Title'].replace('Ms', 'Miss')

train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')

test_df['Title'] = test_df['Title'].replace('Mlle', 'Miss')

test_df['Title'] = test_df['Title'].replace('Ms', 'Miss')

test_df['Title'] = test_df['Title'].replace('Mme', 'Mrs')

#convert title to ordinal

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 0}

combine = [train_df, test_df]

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)
#Age

#Complete missing data, Age

guess_ages = np.zeros((2,3))

guess_ages

for dataset in combine:

    for i in range(0,2):

        for j in range(0,3):

            #use median age of people with same sex and pclass value

            guess_df = dataset[(dataset['Sex']==i)&(dataset['Pclass']==j+1)]['Age'].dropna()

            guess_ages[i,j] = guess_df.median()       

    for i in range(0,2):

        for j in range(0,3):

            dataset.loc[(dataset.Age.isnull())&(dataset.Sex==i)&(dataset.Pclass==j+1), 'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

#Binning Age and convert Age to ordinals

bin = 10

for dataset in combine:

    for i in range(0,int(100/bin)):

        dataset.loc[(dataset['Age']>bin*i)&(dataset['Age']<=bin*i+bin), 'Age'] = i

train_df['Age'].value_counts()
train_df[['Age','Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Survived', ascending=False)


for dataset in combine:

    dataset.loc[dataset['Age'] == 1, 'Age'] = 2

    dataset.loc[dataset['Age'] == 3, 'Age'] = 1

    dataset.loc[dataset['Age'] == 4, 'Age'] = 1

    dataset.loc[dataset['Age'] == 5, 'Age'] = 1

    dataset.loc[dataset['Age'] == 6, 'Age'] = 3

    dataset.loc[dataset['Age'] == 7, 'Age'] = 3

train_df[['Age','Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Survived', ascending=False)    
print (train_df['Title'].value_counts())

print (test_df['Title'].value_counts())
train_df.corr()
#separate train_df to train_train and train_test

train_train = train_df.sample(frac=0.7, random_state=1)

train_test = train_df.loc[~train_df.index.isin(train_train.index)]

print (train_train.shape, train_test.shape)

columns = ['Pclass', 'Sex', 'Title', 'Age']

target = "Survived"
#random forest

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

random_forest = RandomForestClassifier(n_estimators=100, random_state=1)

random_forest.fit(train_train[columns], train_train[target])

print ('train score:', random_forest.score(train_train[columns], train_train[target]))

y_pred = random_forest.predict(train_test[columns])

y_true = train_test[target]

score = accuracy_score(y_true, y_pred)

print ('test score:', score)

#random_forest.score(x_train, y_train)

#acc_random_forest = round(random_forest.score(x_train, y_train)*100, 2)

#acc_random_forest
#random forest on x_test

#from sklearn.ensemble import RandomForestClassifier

#random_forest = RandomForestClassifier(n_estimators=100, random_state=15232)

random_forest.fit(train_df[columns], train_df[target])

y_pred = random_forest.predict(train_df[columns])

print (accuracy_score(train_df[target], y_pred))

random_forest.score(train_df[columns], train_df[target])
print (columns)
#submission

y_pred = random_forest.predict(test_df[columns])

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": y_pred

    })

submission.to_csv('linden_Titanic_submission_sex_pclass_title_age.csv', index=False)