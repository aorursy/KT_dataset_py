import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
import os
os.getcwd()
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train.head()
test.head()
type(train)
train.dtypes
test.dtypes
train.shape
test.shape
train.columns
pd.isna(train).sum()
pd.isna(test).sum()
train.describe()
train['Embarked'].fillna('S', inplace = True)
test['Embarked'].fillna(test['Embarked'].mode(), inplace = True)
train['Age'].fillna(train['Age'].mean(), inplace = True)
test['Age'].fillna(test['Age'].mean(), inplace = True)
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
pd.isna(train['Embarked']).sum()
pd.isna(train['Embarked']).sum()
train.columns
lb = LabelEncoder()
train['Embarked'] = lb.fit_transform(train['Embarked'].astype(str))
train['Sex'] = lb.fit_transform(train['Sex'].astype(str))
test['Embarked'] = lb.fit_transform(test['Embarked'].astype(str))
test['Sex'] = lb.fit_transform(test['Sex'].astype(str))
train_x = train[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
test_1 = test[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
train_x.head()
train_y = train[["Survived"]]
test.head()
model = DecisionTreeClassifier()
model.fit(train_x,train_y)
pred = model.predict(test_1)
result = pd.DataFrame({"PassengerId":test["PassengerId"], "Survived":pred})
print(result)
result.to_csv("Submission2.csv", index=False)