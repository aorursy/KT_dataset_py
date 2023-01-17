import pandas as pd

import numpy as np

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder
#dropped the columns which are not much related to survival probability of passenger

train = pd.read_csv("../input/titanic/train.csv")

train = train.drop(columns = ['Cabin' , 'Name', 'SibSp', 'Ticket', 'Parch'])
#found count of null values of each column

print(train.isnull().sum())
#used simple imputer to fill null values

mean_imputer = SimpleImputer(strategy='mean')

train[['Age']] = mean_imputer.fit_transform(train[['Age']])

constant_imputer = SimpleImputer(strategy='constant', fill_value = "S")

train[['Embarked']] = constant_imputer.fit_transform(train[['Embarked']])

train.head()
print(train['Embarked'].value_counts())
#Changed sex data to label encoded form

le = LabelEncoder()

train['Sex'] = le.fit_transform(train["Sex"])

train.head()
#value_counts of Pclass

print(train['Pclass'].value_counts())
#changed embarked and pclass to one hot encoded form

ohe = pd.get_dummies(train['Embarked'], prefix='Embarked')

train = pd.concat([train, ohe], axis=1)

ohe2 = pd.get_dummies(train['Pclass'], prefix='Pclass')

train = pd.concat([train, ohe2], axis=1)

train = train.drop(columns = ['Embarked', 'Pclass'])

train.head()
#used RandomForestClassifier as our model

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

features = [ 'Sex' ,'Age','Fare' ,'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Pclass_1', 'Pclass_2', 'Pclass_3' ]

rf.fit(X=train[features], y=train['Survived'])
#applied feature engineering to test dataframe

test = pd.read_csv("../input/titanic/test.csv")

test = test.drop(columns = ['Cabin' , 'Name', 'Ticket', 'SibSp', 'Parch'])

from sklearn.impute import SimpleImputer

mean_imputer = SimpleImputer(strategy='mean')

test[['Age']] = mean_imputer.fit_transform(test[['Age']])

constant_imputer = SimpleImputer(strategy='constant', fill_value = "S")

test[['Embarked']] = constant_imputer.fit_transform(test[['Embarked']])

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

test['Sex'] = le.fit_transform(test["Sex"])

ohe3 = pd.get_dummies(test['Embarked'], prefix='Embarked')

test = pd.concat([test, ohe3], axis=1)

ohe4 = pd.get_dummies(test['Pclass'], prefix='Pclass')

test = pd.concat([test, ohe4], axis=1)

test = test.drop(columns = ['Embarked', 'Pclass'])

test.head()
#Checked if there is null values

print(test.isnull().sum())
#filled null values

from sklearn.impute import SimpleImputer

mean_imputer = SimpleImputer(strategy='mean')

test[['Fare']] = mean_imputer.fit_transform(test[['Fare']])

test.head()
#made final prediction on test dataset to kaggle_submission.csv

test['Survived'] = rf.predict(test[features])

test[['PassengerId', 'Survived']].to_csv('titanic_submission.csv', index=False)