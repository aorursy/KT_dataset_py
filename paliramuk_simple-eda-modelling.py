import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib

import numpy

import csv

import os
import os

print(os.listdir("../input"))

titanic_train=pd.read_csv('../input/train.csv')

titanic_test=pd.read_csv('../input/test.csv')

sample_submission=pd.read_csv('../input/gender_submission.csv')
titanic_train.shape, titanic_test.shape,sample_submission.shape
titanic_train.head()
titanic_train.isna().sum()
titanic_test.isna().sum()
titanic_train.dtypes
titanic_train.Sex.replace('female', '0',inplace = True)

titanic_train.Sex.replace('male', '1',inplace = True)

titanic_test.Sex.replace('female', '0',inplace = True)

titanic_test.Sex.replace('male', '1',inplace = True)
titanic_train['Sex'].unique(),titanic_test['Sex'].unique()
titanic_train = titanic_train.drop(['Name','Ticket','Cabin','Embarked'],axis=1)

titanic_test = titanic_test.drop(['Name','Ticket','Cabin','Embarked'],axis=1)
# Replace NaN in Age with median Value

titanic_train = titanic_train.replace(np.nan, 30)

titanic_test = titanic_test.replace(np.nan, 30)
#Convert Fare & Age datatype to Interger

titanic_train['Fare'] = titanic_train['Fare'].astype(np.int64)

titanic_test['Fare'] = titanic_test['Fare'].astype(np.int64)

titanic_train['Age'] = titanic_train['Age'].astype(np.int64)

titanic_test['Age'] = titanic_test['Age'].astype(np.int64)
titanic_train.head()
titanic_train['Survived'].value_counts().plot.bar();
sns.distplot(titanic_train['Fare'], kde=False,bins=10)
sns.barplot(y='Fare',x='Survived',data=titanic_train)
from sklearn.model_selection import train_test_split



predictors = titanic_train.drop(['Survived', 'PassengerId'], axis=1)

target = titanic_train["Survived"]

x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.30, random_state = 0)
# Logistic Regression

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

acc_logreg = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_logreg)
# Random Forest

from sklearn.ensemble import RandomForestClassifier



randomforest = RandomForestClassifier()

randomforest.fit(x_train, y_train)

y_pred = randomforest.predict(x_test)

acc_randomforest = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_randomforest)
#set ids as PassengerId and predict survival 

ids = titanic_test['PassengerId']

predictions = randomforest.predict(titanic_test.drop('PassengerId', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)