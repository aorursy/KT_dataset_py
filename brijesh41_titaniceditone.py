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
train_csv = pd.read_csv('/kaggle/input/titanic/train.csv')

test_csv = pd.read_csv('/kaggle/input/titanic/test.csv')
train_csv = train_csv.drop(['Name','PassengerId','Ticket','Cabin'],axis = 1)

test_csv = test_csv.drop(['Name','PassengerId','Ticket','Cabin'],axis = 1)
train_csv = train_csv[pd.notnull(train_csv['Embarked'])]

test_csv= test_csv[pd.notnull(test_csv['Embarked'])]
train_csv_dummy_s = train_csv['Sex'].str.get_dummies(" ")

test_csv_dummy_s = test_csv['Sex'].str.get_dummies(" ")
print(test_csv_dummy_s.head())

train_csv_dummy_e = train_csv['Embarked'].str.get_dummies(" ")

test_csv_dummy_e = test_csv['Embarked'].str.get_dummies(" ")
print(test_csv_dummy_e.head())
train_csv['index'] = range(0, len(train_csv))

test_csv['index'] = range(0,len(test_csv))
train_csv_dummy_s['index'] = range(0,len(train_csv))

train_csv_dummy_e['index'] = range(0,len(train_csv))

test_csv_dummy_s['index'] = range(0,len(test_csv))

test_csv_dummy_e['index'] = range(0,len(test_csv))
test_csv.columns
train_csv_ = pd.merge(train_csv,train_csv_dummy_s,how='inner',on = 'index')

train_csv_ = pd.merge(train_csv_dummy_e,train_csv_,how= 'inner',on= 'index')

test_csv_ = pd.merge(test_csv,test_csv_dummy_s,how='inner',on = 'index')

test_csv_ = pd.merge(test_csv_dummy_e,test_csv_,how= 'inner',on= 'index')
print(test_csv_.columns)

print(train_csv_.columns)
train_csv_ = train_csv_.drop(['Embarked'],axis = 1)

test_csv_ =  test_csv_.drop(['Embarked'],axis = 1)
print(test_csv_.columns)

print(train_csv_.columns)


train_csv_.Age =  train_csv_.Age.fillna(train_csv.Age.mean())

test_csv_.Age =  test_csv_.Age.fillna(test_csv.Age.mean())
labels = train_csv_.Survived
train_csv_ = train_csv_.drop(['Survived','Sex'],axis =1 )

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

x_train,y_train,x_test,y_test= train_test_split(train_csv_,labels, test_size = 0.1)
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score

gbk = GradientBoostingClassifier()

gbk.fit(x_train, x_test)

y_pred = gbk.predict(y_train)

acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_gbk)
submission_csv = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
submission_csv.columns
test_csv_.columns
test_csv_ = test_csv_.drop(['Sex'],axis = 1)
test_csv_.Fare = test_csv_.Fare.fillna(test_csv_.Fare.mean())
test_csv_.head()
print(test_csv_.columns)

print(train_csv_.columns)
prediction = gbk.predict(test_csv_)
len(prediction)
submission_csv.Survived = prediction
submission_csv.to_csv('submission.csv',index = False)