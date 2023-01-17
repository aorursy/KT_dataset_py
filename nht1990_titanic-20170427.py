import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

# Import the linear regression class

from sklearn.linear_model import LinearRegression

# Sklearn also has a helper that makes it easy to do cross-validation

from sklearn.cross_validation import KFold
titanic=pd.read_csv('../input/train.csv')

titanic.describe()
titanic.head()
titanic['Embarked'].unique()
titanic['Age']=titanic['Age'].fillna(titanic['Age'].median())

titanic.describe()
titanic['Sex'].unique()
titanic.loc[titanic['Sex']=='male','Sex']=0

titanic.loc[titanic['Sex']=='female','Sex']=1
import collections

collections.Counter(titanic['Embarked'])
titanic['Embarked']=titanic['Embarked'].fillna('S')

titanic.loc[titanic['Embarked']=='S','Embarked']=0

titanic.loc[titanic['Embarked']=='C','Embarked']=1

titanic.loc[titanic['Embarked']=='Q','Embarked']=2
titanic.head()
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

x_titanic=titanic[predictors]

y_titanic=titanic['Survived']

linreg = LinearRegression()

linreg.fit(x_titanic,y_titanic)

round(linreg.score(x_titanic, y_titanic) * 100, 2)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(x_titanic,y_titanic)

round(logreg.score(x_titanic, y_titanic) * 100, 2)
test=pd.read_csv('../input/test.csv')

test.head()

test.describe()
test['Age'].unique()

test['Embarked'].unique()
test['Age']=test['Age'].fillna(titanic['Age'].median())

test.loc[test['Sex']=='male','Sex']=0

test.loc[test['Sex']=='female','Sex']=1

test.loc[test['Embarked']=='S','Embarked']=0

test.loc[test['Embarked']=='C','Embarked']=1

test.loc[test['Embarked']=='Q','Embarked']=2

test['Fare']=test['Fare'].fillna(test['Fare'].median())
x_test=test[predictors]

submission=logreg.predict(x_test)

submission=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':submission})

submission.to_csv('submission.csv',index=False)