# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns


%matplotlib inline
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')

train.head()
train.describe(include="all")
#check for any other unusable values
print(pd.isnull(train).sum())
sns.heatmap(train.isnull(), cbar=False)
#draw a bar plot of survival by sex
sns.barplot(x="Sex", y="Survived", data=train)
#drop the name feature since it contains no more useful information.
train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)
# Deleting the two values which are not have an Embarked

train = train.dropna(axis=0, subset=['Embarked'])
test = test.dropna(axis=0, subset=['Embarked'])

# train = train.drop(['Fare'], axis=1)
# test= test.drop(['Fare'], axis=1)

train
#we'll start off by dropping the Cabin feature since not a lot more useful information can be extracted from it.
train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)

#we can also drop the Ticket feature since it's unlikely to yield any useful information
train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)

train.head()
# distribution of the age survival
train["Age"] = train["Age"].fillna(train['Age'].median())
test["Age"] = test["Age"].fillna(test['Age'].median())

train["Fare"].fillna(test["Fare"].median(), inplace=True)
test["Fare"].fillna(test["Fare"].median(), inplace=True)

train

#check for any other unusable values
print(pd.isnull(train).sum())
print("-"*40)
print(pd.isnull(test).sum())
print(train.head())
# One hot encoding!
out1 = pd.get_dummies(train['Embarked'], prefix='Embarked')
out1_test = pd.get_dummies(test['Embarked'], prefix='Embarked')

train = pd.concat([train, out1], axis=1)
test = pd.concat([test, out1_test], axis=1)


test
#map each Sex value to a numerical value
sex_encoding = pd.get_dummies(train['Sex'])
sex_encoding_test = pd.get_dummies(test['Sex'])

train = pd.concat([train, sex_encoding], axis=1)
test = pd.concat([test, sex_encoding_test], axis=1)

train.head()
# Haha this is making my prediction worse. Needs some adjustments probably
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# scaled_age_train = scaler.fit_transform(train_age)
# scaled_age_test = scaler.fit_transform(test_age)

# train['ScaledAge'] = scaled_age_train
# test['ScaledAge'] = scaled_age_test

train
print('='*50)
print("Number of columns in training data")
print('='*50)
print("\n")
print(train.columns.values)
print("\n")
print('='*50)
print("Number of columns in test data")
print('='*50)
print("\n")
print(test.columns.values)
#splitting the predictors from the targets
from sklearn.model_selection import train_test_split

predictors = train.drop(['Survived', 'PassengerId', 'Embarked', 'Sex'], axis=1)
test_features = test.drop(['PassengerId', 'Embarked', 'Sex'], axis=1)

target = train['Survived']

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
x_train.shape, x_val.shape, y_train.shape, y_val.shape
#predicting
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier()
random_forest.fit(x_train, y_train)

y_pred = random_forest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)

# submission
pred_test = random_forest.predict(test_features)

output = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_test})
output.to_csv('submission.csv', index=False)