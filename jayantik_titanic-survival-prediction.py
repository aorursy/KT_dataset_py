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
 # Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
print(os.getcwd()) #know the current working dir
os.chdir("../input/") #set working dir
print(os.getcwd())
# Importing the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.info()
test.columns
# drop unnecessary columns, these columns won't be useful in analysis and prediction
train = train.drop(['PassengerId','Name','Ticket'], axis=1)
test    = test.drop(['Name','Ticket'], axis=1)
train.info()
test.info()
# Data Preprocessing
#Embarked # only in train, fill the two missing values with the most occurred value, which is "S".
train["Embarked"] = train["Embarked"].fillna("S")
embark_dummies_titanic  = pd.get_dummies(train['Embarked'])
embark_dummies_titanic.drop(['S'], axis=1, inplace=True)

embark_dummies_test  = pd.get_dummies(test['Embarked'])
embark_dummies_test.drop(['S'], axis=1, inplace=True)

train = train.join(embark_dummies_titanic)
test    = test.join(embark_dummies_test)

train.drop(['Embarked'], axis=1,inplace=True)
test.drop(['Embarked'], axis=1,inplace=True)
train.info()
# Fare # only for test_df, since there is a missing "Fare" values
test["Fare"].fillna(test["Fare"].median(), inplace=True)

# convert from float to int
train['Fare'] = train['Fare'].astype(int)
test['Fare']    = test['Fare'].astype(int)
train.info()
test.info()
# Age #  fill NaN values in Age column with random values generated
train["Age"][np.isnan(train["Age"])]
train["Age"].fillna(train["Age"].mean(), inplace =True)
test["Age"][np.isnan(test["Age"])]
test["Age"].fillna(test['Age'].mean(), inplace = True)
# convert from float to int
train['Age'] = train['Age'].astype(int)
test['Age']  = test['Age'].astype(int)
train.info()
test.info()
# Encoding categorical data
sex = pd.get_dummies(train['Sex'], drop_first=True, prefix='sex')
pclass = pd.get_dummies(train['Pclass'], drop_first=True, prefix='Pclass')
train = pd.concat([train, sex, pclass], axis=1)
sex1 = pd.get_dummies(test['Sex'], drop_first=True, prefix='sex')
pclass1 = pd.get_dummies(test['Pclass'], drop_first=True, prefix='Pclass')

test = pd.concat([test, sex1, pclass1], axis=1)
train.drop(['Pclass','Sex'] , axis=1, inplace=True)
train.head()

test.drop(['Pclass','Sex'] , axis=1, inplace=True)
test.head()

# Cabin has a lot of NaN values, so it won't cause a remarkable impact on prediction
train.drop("Cabin",axis=1,inplace=True)
test.drop("Cabin",axis=1,inplace=True)
train.info()
test.info()
# define training and testing sets
X = train.drop("Survived",axis=1)
Y = train["Survived"]
X_test  = test.drop("PassengerId",axis=1).copy()
# Machine Learning # Fitting Logistic Regression to training set
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, Y)
# Predicting the values
y_pred = model.predict(X)
model.score(X, Y)
