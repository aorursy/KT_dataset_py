# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
print('Hello World!')
print('Hello kaggle!')
train=pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
test=pd.read_csv('/kaggle/input/titanic/test.csv')



test.head()
def graph(feature):

    survived=train[train['Survived']==1][feature].value_counts()

    dead=train[train['Survived']==0][feature].value_counts()

    df=pd.DataFrame([survived,dead])

    df.index=['survived','dead']

    df.plot.pie(subplots=True)
graph('Sex')
graph('Pclass')
graph('Embarked')
#乗客番号,名前,チケット,部屋番号を削除

#testの方の乗客番号を残すのは、提出時のフォーマットに必要なため

train = train.drop(['PassengerId','Name','Ticket','Cabin','Embarked'], axis=1)

test = test.drop(['Name','Ticket','Cabin','Embarked'], axis=1)
#学習データの欠損値を確認

train.isnull().sum()
#テストデータの欠損値を確認

test.isnull().sum()
#年齢（'Age'）の欠損値を、ほかの年齢の平均で埋める

train['Age'].fillna(train['Age'].mean(),inplace=True)

test['Age'].fillna(test['Age'].mean(),inplace=True)

#テスト用データの運賃('Fare')の欠損値を、ほかの運賃の平均で埋める

test['Fare'].fillna(test['Fare'].mean(),inplace=True)
train['Sex'] = train['Sex'].replace(['female','male'],[0,1])

train.head()
test['Sex'] = test['Sex'].replace(['female','male'],[0,1])

test.head()
X_train = train.drop("Survived", axis=1)

Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
from sklearn.ensemble import RandomForestClassifier as RandomForest
model = RandomForest(n_estimators=1000,bootstrap=True)

model.fit(X_train, Y_train)
result = model.predict(X_test)
#  提出する

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": result

    })

#一日１０回までなのでご注意ください。

submission.to_csv("submission.csv", index=False)