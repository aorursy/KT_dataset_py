import pandas as pd
import numpy as np
import csv

from matplotlib import pyplot as plt
import seaborn as ans
%matplotlib inline

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

train = pd.read_csv("../titanic/train.csv")
test = pd.read_csv("../titanic/test.csv")
train.head()
train["Sex"] = train["Sex"].astype('category')
train["Sex"] = train["Sex"].cat.codes

test["Sex"] = test["Sex"].astype('category')
test["Sex"] = test["Sex"].cat.codes


train['FamilySize'] = train['SibSp'] + train['Parch']
train.head()
train.drop('SibSp',axis=1,inplace=True)
train.drop('Parch',axis=1,inplace=True)
train.drop('Name',axis=1,inplace=True)
train.drop('Ticket',axis=1,inplace=True)
train.drop('Cabin',axis=1,inplace=True)
train.drop('Embarked', axis=1, inplace=True)
train = pd.get_dummies(train, drop_first=True)
traintrain, testtrain = train_test_split(train, test_size=0.2)
train1 = train1.fillna(train.median())
test = test.fillna(train.median())

test['FamilySize'] = test['SibSp'] + test['Parch']
test.drop('SibSp',axis=1,inplace=True)
test.drop('Parch',axis=1,inplace=True)
x = train1["Survived"]
y = train1[["Pclass", "Sex", "Age", "Fare", "FamilySize"]]
model = LogisticRegression()

model.fit(y, x)
z = test[["Pclass", "Sex", "Age", "Fare", "FamilySize"]]
pred = model.predict(z)
d = pd.DataFrame({'PassengerId': test['PassengerId'].values, 'Survived': pred})
d.to_csv('outt.csv', index=False)


