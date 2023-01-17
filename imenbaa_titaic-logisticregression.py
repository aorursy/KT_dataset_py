# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt



gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")

train
train['Age'] = train['Age'].fillna(value=train['Age'].mean())

train['Fare'] = train['Fare'].fillna(value=train['Fare'].mean())

train["Embarked"]=np.where(train["Embarked"]=="S",0,np.where(train["Embarked"]=="C",1,2))

train["Sex"]=np.where(train["Sex"]=="male",0,1)

train
test['Age'] = test['Age'].fillna(value=test['Age'].mean())

test['Fare'] = test['Fare'].fillna(value=test['Fare'].mean())

test["Embarked"]=np.where(test["Embarked"]=="S",0,np.where(test["Embarked"]=="C",1,2))

test["Sex"]=np.where(test["Sex"]=="male",0,1)
test
data_train=train[["PassengerId","Survived","Pclass","Sex","Age","Embarked"]]

data_train=data_train.dropna()

y=data_train["Survived"]

X=data_train[["PassengerId","Pclass","Sex","Age","Embarked"]]

X
data_test=test[["PassengerId","Pclass","Sex","Age","Embarked"]]

data_test=data_test.dropna()

data_test
from sklearn.linear_model import LogisticRegression

model=LogisticRegression(C=10)

model.fit(X,y)
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(data_test)
roc_auc_score(gender_submission["Survived"], y_pred)