# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))
train_path = '../input/train.csv'
input_data = pd.read_csv(train_path)
# print(input_data.head())
input_data.head()
# Any results you write to the current directory are saved as output.
input_data.loc[(input_data.Cabin.notnull()),'Cabin']=True
input_data.loc[(input_data.Cabin.isnull()),'Cabin']=False
# input_data.groupby(['Cabin'])['Survived'].mean().plot(kind='bar')
input_data['Age'].fillna((input_data['Age'].mean()),inplace=True)
# input_data.sample(10)
# np.random.seed(0)
input_data.sample(10)

# print(input_data.shape)
input_data['Embarked'].value_counts()
input_data['Embarked'].fillna('S',inplace=True)

# get the number of missing data points per column
missing_values_count = input_data.isnull().sum()
# look at the # of missing points
print(missing_values_count)
y = input_data.Survived
X = input_data.drop(['PassengerId','Survived','Name','Ticket'],axis = 1)

X = pd.get_dummies(X)

test_path = '../input/test.csv'
test_data = pd.read_csv(test_path)
test_data.loc[(input_data.Cabin.notnull()),'Cabin']=True
test_data.loc[(input_data.Cabin.isnull()),'Cabin']=False

test_data['Age'].fillna((input_data['Age'].mean()),inplace=True)

# print(test_data.shape)
print(test_data['Pclass'][test_data['Fare'].isnull()])
# print(test_data['Pclass'][152])
test_data['Fare'][test_data.Pclass.values==3].mean()
test_data['Fare'].fillna(test_data['Fare'][test_data.Pclass.values==3].mean(),inplace=True)

missing_values_count2 = test_data.isnull().sum()
print(missing_values_count2)
test_X = test_data.drop(['PassengerId','Name','Ticket'],axis = 1)

test_X = pd.get_dummies(test_X)

print(test_X.isnull().sum())
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf=rf.fit(X,y)
test_result=rf.predict(test_X)
test_result
result=pd.DataFrame({ 'PassengerId': test_data.PassengerId, 'Survived': test_result })
result.head()
result.to_csv("Titanic_result.csv", index=False)