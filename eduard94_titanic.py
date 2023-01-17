import numpy as np

import pandas as pd

import os
test_data = pd.read_csv("../input/test.csv")

train_data = pd.read_csv("../input/train.csv")

gender_submission = pd.read_csv("../input/gender_submission.csv")

train_data.head()
train_data = train_data.drop(['Name', 'Ticket', 'PassengerId'], 1)

test_data = test_data.drop(['Name', 'Ticket', 'PassengerId'], 1)
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)

test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)

test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)



train_data['Cabin'].fillna(0, inplace=True)

test_data['Cabin'].fillna(0, inplace=True)



train_data['Cabin'] = [1 if x != 0 else x for x in train_data['Cabin']]

test_data['Cabin'] = [1 if x != 0 else x for x in test_data['Cabin']]
train_Pclass = pd.get_dummies(train_data['Pclass'])

test_Pclass = pd.get_dummies(test_data['Pclass'])



train_Sex = pd.get_dummies(train_data['Sex'])

test_Sex = pd.get_dummies(test_data['Sex'])



train_Embarked = pd.get_dummies(train_data['Embarked'])

test_Embarked = pd.get_dummies(test_data['Embarked'])
train_data = pd.concat([train_data, train_Pclass, train_Sex, train_Embarked], axis=1)

test_data = pd.concat([test_data, test_Pclass, test_Sex, test_Embarked], axis=1)
train_data = train_data.drop(['Pclass', 'Sex', 'Embarked'], 1)

test_data = test_data.drop([ 'Pclass', 'Sex', 'Embarked'], 1)

train_data.head()
x = train_data.drop(['Survived'], 1)

y = train_data['Survived']
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state=2)
from sklearn.linear_model import LogisticRegression



model = LogisticRegression()



model.fit(x_train, y_train)

model.score(x_test, y_test)
prediction = model.predict(test_data)

gender_submission['Survived'] = prediction

gender_submission.to_csv("submit.csv", index=False)