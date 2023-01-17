# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

test = pd.read_csv("../input/titanic-machine-learning-from-disaster/test.csv")

train = pd.read_csv("../input/titanic-machine-learning-from-disaster/train.csv")
test.head(5)
train.head(5)
test.shape
train.shape

len(train)
train_required = train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Survived','PassengerId']]

train_required.head(5)

len(train_required)

train_required.isnull()
train_required['Embarked'].isnull().values.any()
train_required['Age'].fillna(train_required['Age'].mean(),inplace=True)
train_required['Age'].mean()
train_required = train_required.drop('Cabin',axis=1)
train_required.head()

train_required = train_required.dropna()
len(train_required)
from sklearn import preprocessing

lblEncoder = preprocessing.LabelEncoder()

train_required['Sex'] = lblEncoder.fit_transform(train_required['Sex'])

train_required['Embarked'] = lblEncoder.fit_transform(train_required['Embarked'])

train_required.head()
from sklearn.naive_bayes import GaussianNB

ngnb =GaussianNB()
X_train = train_required.drop('Survived',axis=1)

Y_train = train_required['Survived']



X_train.head(5)

Y_train.head(5)
ngnb.fit(X_train,Y_train)


test = pd.read_csv("../input/titanic-machine-learning-from-disaster/test.csv")

test.head(5)
test = test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','PassengerId']]

test.head(5)
test['Embarked'].isnull().values.any()
test['Age'].fillna(test['Age'].mean(),inplace=True)
len(test)
test_req = test.dropna()

len(test_req)
from sklearn import preprocessing

lblEncoderTest = preprocessing.LabelEncoder()

test_req['Sex'] = lblEncoderTest.fit_transform(test_req['Sex'])

test_req['Embarked'] = lblEncoderTest.fit_transform(test_req['Embarked'])

test_req.head()
y_pred = ngnb.predict(test_req)
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred)
submission = pd.DataFrame({"PassengerId": test_req["PassengerId"], "Survived": y_pred})



submission.to_csv("Titanic Predictions 1.csv", index=False)



submission.head()