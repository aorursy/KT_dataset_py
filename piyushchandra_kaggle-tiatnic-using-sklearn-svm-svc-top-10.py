# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.head(10)
test.head(10)
train.plot(kind='scatter',x='Survived',y='Age')

plt.show()
train.plot(kind='bar',x='Survived',y='Pclass')

plt.show()
train.plot(kind='bar',x='Survived',y='Fare')

plt.show()
train.plot(kind='bar',x='Survived',y='Parch')

plt.show()
x_train=train.drop("Survived",axis=1)

y_train=train["Survived"]

x_test=test
x_train=x_train.drop('PassengerId',axis=1)

x_train=x_train.drop('Name',axis=1)

x_train=x_train.drop('Ticket',axis=1)

x_train=x_train.drop('Cabin',axis=1)
x_test=x_test.drop('Name',axis=1)

x_test=x_test.drop('PassengerId',axis=1)

x_test=x_test.drop('Ticket',axis=1)

x_test=x_test.drop('Cabin',axis=1)
x_train=pd.get_dummies(x_train)

x_test=pd.get_dummies(x_test)
x_train.head(10)
x_test.head(10)
x_train.info()
x_test.info()
x_train.Age.fillna(x_train.Age.mean(), inplace=True)

x_test.Age.fillna(x_test.Age.mean(), inplace=True)

x_test.Fare.fillna(method='bfill',inplace=True)
train_stats=x_train.describe()

train_stats=train_stats.transpose()



def norm_train(x):

  return (x - train_stats['mean']) / train_stats['std']

x_train=norm_train(x_train)

x_test=norm_train(x_test)
x_train.head(20)
x_test.head(10)
from sklearn.svm import SVC

clf=SVC(C=0.5,gamma='auto')

clf.fit(x_train, y_train)

pred=clf.predict(x_test)
acc = round(clf.score(x_train, y_train) * 100, 3)

acc
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": pred

    })

submission.to_csv("titanic.svm.SVC.csv")