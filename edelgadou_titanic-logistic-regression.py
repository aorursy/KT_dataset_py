import pandas as pd
import numpy as np 

from matplotlib import pyplot as git
import seaborn as sns
%matplotlib inline

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
import os 

os.path.realpath('.')


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train[['Sex', 'Survived']].head()
train.head()
train.describe()
pd.isna(train).sum()
#how many values are missing
train['Age'].fillna(train['Age'].mean(), inplace=True)
del train['Cabin']
del train['Embarked']
test['Age'].fillna(test['Age'].mean(), inplace=True)
test['Fare'].fillna(test['Fare'].mean(), inplace=True)
del test['Cabin']
del test['Embarked']
pd.isna(train).sum()
pd.isna(test).sum()
train = pd.get_dummies(train, columns=['Sex'])
test = pd.get_dummies(test, columns=['Sex'])
y = train['Survived']
x = train[['Pclass', 'Age', 'SibSp', 'Fare', 'Parch', 'Sex_female', 'Sex_male']]
model = LogisticRegression()
fit = model.fit(x, y)
z = test[['Pclass', 'Age', 'SibSp', 'Fare', 'Parch', 'Sex_female', 'Sex_male']]
Survived = model.predict(z)
Survived
Passenger = test['PassengerId']
Submission = pd.DataFrame({"PassengerId" : Passenger, "Survived" : Survived})
Submission.to_csv("submission1.csv", index=False)
fit.score(x, y)
