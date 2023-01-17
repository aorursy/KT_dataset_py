# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import math
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
train.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)

test.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
train = pd.get_dummies(train, columns=['Pclass','Sex','Embarked'],drop_first=True)
test = pd.get_dummies(test, columns=['Pclass','Sex','Embarked'],drop_first=True)
train.isnull().sum()
X_train = train.drop('Survived',axis=1)
Y_train = train['Survived']

X_train = sm.add_constant(X_train)
xmedian = X_train['Age'].median()
xmedian
X_train['Age'].fillna(xmedian, inplace=True)
X_train.Age.isnull().sum()
test.head()
test.isnull().sum()
xmedian = test['Age'].median()
xmedian
test['Age'].fillna(xmedian, inplace=True)
test.Age.isnull().sum()
logit = sm.GLM(Y_train,X_train, family=sm.families.Binomial())
full_model = logit.fit()
print(full_model.summary2())
X_train.drop(["Embarked_Q", "Embarked_S"], axis=1, inplace=True)
print(sm.GLM(Y_train,X_train, family=sm.families.Binomial()).fit().summary2())
X_train.drop(['Parch'], axis=1, inplace=True)
print(sm.GLM(Y_train,X_train, family=sm.families.Binomial()).fit().summary2())
X_train.drop("Fare", axis=1, inplace=True)
print(sm.GLM(Y_train,X_train, family=sm.families.Binomial()).fit().summary2())
finalmodel = sm.GLM(Y_train,X_train,family=sm.families.Binomial()).fit()
np.exp(finalmodel.params)
finalmodel.aic
finalmodel.deviance
test = sm.add_constant(test[['Age','SibSp','Pclass_2','Pclass_3','Sex_male']])
test.head(5)
probabilities = finalmodel.predict(test)
predicted_class=probabilities.map(lambda x:1 if x > 0.5 else 0)
predicted_class.head(5)
from sklearn.metrics import accuracy_score
