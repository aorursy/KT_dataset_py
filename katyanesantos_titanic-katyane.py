#Bibliotecas utilizadas

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import time

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

%matplotlib inline
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train.sample(5)
plt.figure(figsize=(9,6))

sns.barplot(x='Sex',y='Survived',data=train)
plt.figure(figsize=(9,6))

sns.barplot(x='Pclass', y='Survived', data=train)
train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
train.isnull().sum()
from cesium import featurize

from sklearn.impute import SimpleImputer 

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
train_numerico = train.drop(['Sex','Embarked'], axis=1)

test_numerico = test.drop(['Sex','Embarked'], axis=1)
imputer.fit(train_numerico)
train_no_null = imputer.transform(train_numerico)
imputer.fit(test_numerico)

test_no_null = imputer.transform(test_numerico)
train_tr = pd.DataFrame(train_no_null, columns=train_numerico.columns)

test_tr = pd.DataFrame(test_no_null, columns=test_numerico.columns)
train_cat = train[['Sex', 'Embarked']]

test_cat = test[['Sex', 'Embarked']]

train_cat_encoded = pd.get_dummies(train_cat)

test_cat_encoded = pd.get_dummies(test_cat)
train_cat_encoded.head()
new_train = train_tr.join(train_cat_encoded)

new_test = test_tr.join(test_cat_encoded)
from sklearn.model_selection import train_test_split



predictors = new_train.drop(['Survived', 'PassengerId'], axis=1)

target = train["Survived"]

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.2, random_state = 0)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

decisiontree = DecisionTreeClassifier()

decisiontree.fit(x_train, y_train)

y_pred = decisiontree.predict(x_val)

acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_decisiontree)
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()

gbk.fit(x_train, y_train)

y_pred = gbk.predict(x_val)

acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gbk)
ids = test['PassengerId']

predictions = gbk.predict(new_test.drop('PassengerId', axis=1))
Titanic_Katyane = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions.astype('int64') })

Titanic_Katyane.to_csv('Titanic_Katyane.csv', index=False)