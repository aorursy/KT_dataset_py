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
test_df.info()
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

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

combine = [train_df, test_df]

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

#Sex

#convert categorical features to numerical

for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map({'male':0,'female':1}).astype(int)

#Title

#Embarked

#replace missing Embarked with mode

train_df.loc[train_df['Embarked'].isnull(), 'Embarked']=train_df.Embarked.dropna().mode()[0]

#convert Embarked to ordinals

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'Q':1, 'C':2}).astype(int)
#create FamilySize based on SibSp and Parch

for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df['FamilySize'].value_counts()

#Fare

#complete mising Fare in test_df

pf=test_df[['Pclass', 'Fare']].groupby('Pclass', as_index=False).median()

test_df.loc[(test_df.Fare.isnull())&(test_df.Pclass==3), 'Fare'] = pf.loc[pf['Pclass']==3, 'Fare'].median()

#adjust Fare based on FamilySize, since Fare is family based

train_df['Fare'] = train_df['Fare']/train_df['FamilySize']

test_df['Fare'] = test_df['Fare']/test_df['FamilySize']

combine=[train_df, test_df]
#fix 0 Fare

'''

print (train_df[['Pclass', 'Fare']].groupby('Pclass').median())

train_df.loc[(train_df.Pclass==1) & (train_df.Fare<1), 'Fare'] = 33

train_df.loc[(train_df.Pclass==2) & (train_df.Fare<1), 'Fare'] = 13

train_df.loc[(train_df.Pclass==3) & (train_df.Fare<1), 'Fare'] = 7

test_df.loc[(test_df.Pclass==1) & (test_df.Fare<1), 'Fare'] = 33

test_df.loc[(test_df.Pclass==2) & (test_df.Fare<1), 'Fare'] = 13

test_df.loc[(test_df.Pclass==3) & (test_df.Fare<1), 'Fare'] = 7

combine=[train_df, test_df]

'''
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
#convert Fare to ordinal

bin, max = 4, 48

for dataset in combine:

    for i in range(0, int(max/bin)):

        dataset.loc[(dataset['Fare'] > bin*i) & (dataset['Fare'] <= bin*i+bin), 'Fare'] = i

    dataset.loc[ dataset['Fare'] > max, 'Fare'] = int(max/bin)

    dataset['Fare'] = dataset['Fare'].astype(int)
train_df['Fare'].value_counts()
#Binning Age and convert Age to ordinals

bin, max = 20, 100

for dataset in combine:

    for i in range(0,int(max/bin)):

        dataset.loc[(dataset['Age']>bin*i)&(dataset['Age']<=bin*i+bin), 'Age'] = i
train_df['Age'].value_counts()
train_df.corr()
print ('1' if True else '2')
for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize']>1, 'IsAlone'] = 1

train_df['IsAlone'].value_counts()
#drop not needed columns, axis=1 denotes column, default axis=0 denotes row

train_df = train_df.drop(['Ticket', 'Cabin', 'Name', 'PassengerId', 'SibSp', 'Parch'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin', 'Name', 'SibSp', 'Parch'], axis=1)

train_df = train_df.drop(['FamilySize'], axis=1)

test_df = test_df.drop(['FamilySize'], axis=1)

combine=[train_df, test_df]
train_df.corr()
#Use Sex, Pclass, Age, Title, Embarked, Fare, FamilySize --accuracy: 88.66

x_train = train_df.drop(['Survived'], axis=1)

y_train = train_df['Survived']

x_test = test_df.drop(['PassengerId'], axis=1).copy()

x_train.shape, y_train.shape, x_test.shape
#separate train_df to train_train and train_test

train_train = train_df.sample(frac=0.7, random_state=1)

train_test = train_df.loc[~train_df.index.isin(train_train.index)]

print (train_train.shape, train_test.shape)

columns = train_train.columns.tolist()

columns = [c for c in columns if c not in ['Survived']]

target = "Survived"

print (columns)
#random forest

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

random_forest = RandomForestClassifier(n_estimators=100, random_state=15232)

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

random_forest.fit(x_train, y_train)

y_pred = random_forest.predict(x_test)

random_forest.score(x_train, y_train)
#submission

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": y_pred

    })

submission.to_csv('linden_Titanic_submission_isalone.csv', index=False)