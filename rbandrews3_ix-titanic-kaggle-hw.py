import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression


import os
os.path.realpath('.')

os.chdir("/Users/rbandrews3/Desktop/iXperience/Day15/data")

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")


train
train.head()

train.columns
group = train[["Pclass", "Survived"]].groupby(["Pclass", "Survived"])
train['Age'].fillna(train["Age"].mean(), inplace = True)
test['Age'].fillna(test["Age"].mean(), inplace = True)
del train["Cabin"]
train.head()
del train["Embarked"]
col = ['Sex']
train[col] = train[col].apply(lambda x: pd.factorize(x)[0] + 1)
test[col] = test[col].apply(lambda x: pd.factorize(x)[0] + 1)

y = train["Survived"]
x = train[['Pclass', 'Age', 'Fare', 'SibSp', 'Sex']]
model = LogisticRegression()
model.fit(x, y)
model.predict(x)
z = test[['Pclass', 'Age', 'Fare', 'SibSp', 'Sex']]
survival = model.predict(z)
survival
pd.isna(test).sum()

test.head()
test[test['Fare'].isnull()]
test['Fare'].fillna(test["Fare"].mean(), inplace = True)
for i in range(0,890):
    if (pd.isna(test['Fare'])):
        print(i)
passenger_ids = test['PassengerId']
submission = pd.DataFrame({'PassengerId' : test['PassengerId'].values, 'Survived' : survival})
submission
submission.drop(submission.columns[i], axis=0)
submission.to_csv("Submission.csv")