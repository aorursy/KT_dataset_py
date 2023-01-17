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
#read the data from input and output files

train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')

combine = [train_df, test_df]

#Drop features we are not going to use

#train_df = train_df.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)

#test_df = test_df.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)



#train_df.head(3)
train_df['Fare'] = train_df['Fare'].fillna(0)

test_df['Fare'] = test_df['Fare'].fillna(0)



target = 'Survived'

#train_df.head(3)
#test_df.head(3)
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
freq_port = train_df.Embarked.dropna().mode()[0]

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
#Fill in missing age values with 0 (presuming they are a baby if they do not have a listed age)

age_port = train_df.Age.dropna().mode()[0]

for dataset in combine:

    dataset['Age'] = dataset['Age'].fillna(age_port)

for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']
#train_df.head()
for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
for dataset in combine:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

#train_df.head()
train_df = train_df.drop(['Name','Ticket', 'Cabin','FamilySize','SibSp','Parch' ],axis=1)

test_df = test_df.drop(['Name','Ticket', 'Cabin','FamilySize','SibSp','Parch' ],axis=1)
#train_df = pd.get_dummies(train_df)

#test_df = pd.get_dummies(test_df)
#Create classifier object with default hyperparameters

clf = RandomForestClassifier()

#clf = DecisionTreeClassifier(max_depth=3,min_samples_leaf=2)





actual = train_df[target]

train_df=train_df.drop('Survived', axis=1)





#Fit our classifier using the training features and the training target values

clf.fit(train_df,actual) 



#Make predictions using the features from the test data set

predictions = clf.predict(test_df)

submission = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':predictions})

#submission.head()
#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'Titanic Predictions 1.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)