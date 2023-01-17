# Importing all the necessary libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

%matplotlib inline



# Machine Learning modules

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier
#Loading our Dataset

df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
#Preparing our Dataset 

warnings.filterwarnings('ignore')

#Cleaning the data and creating new features

#Age column has NaN values which will be replaced by mean value of age

df.fillna({

    'Age' : df['Age'].median(),

    'Embarked' : 'S',

}, inplace = True)

test_df.fillna({

    'Age' : test_df['Age'].median(),

    'Embarked' : 'S', 

    'Fare' : test_df['Fare'].median()

}, inplace = True)



# We dont need columns like 'PassengerID' & 'Ticket' since it doesnt affect the outcome

# We are also droppping the value of 'Cabin' due to large missing values

df = df.drop(['PassengerId','Ticket','Cabin'], axis=1)

test_df = test_df.drop(['Ticket','Cabin'], axis=1)

df.head(10)
#Pclass

#from observations it is found that survival chance of class 1, class 2 and class 3 are 0.62, 0.47 and 0.24 respectively

#These Observations were found by applying filters in excel sheet for analysis purpose

#Though survival rate for class 3 is very less, but it can't be neglected

#Or we can get the above data through following codes 

df[['Pclass', 'Survived']].groupby('Pclass', as_index=False).mean()

fig,(ax1,ax2) = plt.subplots(1, 2, figsize=(15,5))

sns.barplot('Pclass','Survived', data=df, ax=ax1)

sns.factorplot('Pclass', 'Survived', 'Sex', data=df, kind='bar', palette='muted', legend=False, ax=ax2)
#Age

grid = sns.FacetGrid(df, col='Survived')

grid.map(plt.hist, 'Age', bins=20)

#This gives us an overall idea of how survival rate depend on age

g = sns.FacetGrid(df, col='Survived', row = 'Pclass',size=2.5, aspect=1.5)

g.map(plt.hist, 'Age', alpha=0.7, bins=20)
#Countinue with Age

fig, axis1 = plt.subplots(1,1,figsize=(30,5))

sns.barplot(df['Age'].astype(int),'Survived', data=df)
#Embarked

#Passangers started their journey from three places denoted as 'S','C' & 'Q'

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))

sns.factorplot('Embarked','Survived','Sex',data=df, ax=ax1)

sns.barplot('Embarked','Survived',data=df,ax=ax2)
# Replacing the string value of Embarked column to integers to avoid value error

df['Embarked'].loc[df.Embarked == 'S']=1

df['Embarked'].loc[df.Embarked == 'C']=2

df['Embarked'].loc[df.Embarked == 'Q']=3

test_df['Embarked'].loc[test_df.Embarked == 'S']=1

test_df['Embarked'].loc[test_df.Embarked == 'C']=2

test_df['Embarked'].loc[test_df.Embarked == 'Q']=3
#Name

#This feature can be dropped or neglected but it has a unique structure that might corelate with survival 

combine = [df,test_df]

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand = False)

pd.crosstab(df['Title'],df['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].str.replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].str.replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].str.replace('Mme', 'Mrs')

    

df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

df.head()
#now we can drop the Name column safely from our dataset

df = df.drop('Name', axis =1)

test_df = test_df.drop('Name',axis=1)
# Replace Male with 0 and Female with 1



df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

test_df['Sex'] = test_df['Sex'].map({'female': 1, 'male':0}).astype(int)
#We have 2 columns SibSp and Parch which explains if a particular passenger was alone or with family

#Lets create a column where we can assign 1 if passenger was with family and 0 if he/she was alone

df['Family'] = df['SibSp']+df['Parch']

df['Family'].loc[df['Family'] > 0] = 1

df['Family'].loc[df['Family'] == 0] = 0



test_df['Family'] = test_df['SibSp'] + test_df['Parch']

test_df['Family'].loc[test_df['Family'] > 0] = 1

test_df['Family'].loc[test_df['Family'] == 0] = 0

# Drop the columns SibSp and Parch

df = df.drop(['SibSp','Parch'], axis = 1)

test_df = test_df.drop(['SibSp','Parch'], axis = 1)

fig, ax1 = plt.subplots(1,1,figsize=(10,5))

sns.countplot('Family', data=test_df, ax = ax1)

ax1.set_xticklabels(['With Family','Alone'], rotation = 0)
# Fare price is written in float and varies from $0 to as high as $512

df['Fare'] = df['Fare'].astype(int)

test_df['Fare'] = test_df['Fare'].astype(int)

sns.barplot('Survived','Fare', data=df)
#Preparing Training and Testing Data

X_train = df.drop('Survived', axis=1)

Y_train = df['Survived']

X_test = test_df.drop('PassengerId',axis=1).copy()
# SVM

svmclf = SVC()

svmclf.fit(X_train, Y_train)

Y_test = svmclf.predict(X_test)

svmclf.score(X_train, Y_train)
#Random Forest 

rfclf = RandomForestClassifier()

rfclf.fit(X_train, Y_train)

Y_test = rfclf.predict(X_test)

rfclf.score(X_train, Y_train)
adclf = AdaBoostClassifier()

adclf.fit(X_train, Y_train)

Y_test = adclf.predict(X_test)

adclf.score(X_train, Y_train)