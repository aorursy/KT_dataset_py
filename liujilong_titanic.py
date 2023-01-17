# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
print(train_data.isnull().sum())

print(train_data.shape)
print(test_data.isnull().sum())

print(test_data.shape)
train_data = train_data.drop('Ticket',1).drop('Name',1).drop('Cabin',1)

train_data
train_data['Sex'] = train_data['Sex'].map({'male':1,'female':0})

train_data
print(train_data.isnull().sum())

print(train_data.shape)
train_data = train_data.drop(train_data.loc[train_data['Embarked'].isnull()].index)
print(train_data.isnull().sum())

print(train_data.shape)
train_data = train_data.drop(train_data.loc[train_data['Age'].isnull()].index)
print(train_data)
print(train_data.isnull().sum())

print(train_data.shape)
print(train_data)
train_data['Embarked2'] = train_data['Embarked'].map({'S':1,'C':1,'Q':0})



print(train_data)
train_data['Embarked'] = train_data['Embarked'].map({'S':0,'C':1,'Q':1})

train_data
train_count =(int) (train_data.shape[0]*3/4)

print(train_count)
train = train_data.iloc[0:train_count,2:train_data.shape[1]]

cv = train_data.iloc[train_count : train_data.shape[0], 2:train_data.shape[1]]
print(train.tail(2))
print(cv.tail(2))
train_y = train_data.iloc[0:train_count,1]

cv_y = train_data.iloc[train_count : train_data.shape[0],1]
train_y_r = train_y.map({1:0,0:1})

train_y_r.name = "NotSurvived"

train_y_score = pd.concat([train_y, train_y_r], axis=1)

print(train_y_score)
cv_y_r = cv_y.map({1:0,0:1})

cv_y_r.name = "NotSurvived"

cv_y_score = pd.concat([cv_y, cv_y_r], axis=1)

print(cv_y_score)
print(train.shape)

print(train_y_score.shape)

print(cv.shape)

print(cv_y_score.shape)
import tensorflow as tf
null
null


from sklearn import svm

clf = svm.SVC()

clf.fit(train, train_y.values.ravel())

clf.score(cv,cv_y)
print(train.head())

print(test_data.head())
test_data = test_data.drop('Name',1).drop('Ticket',1)
test_data = test_data.drop('Cabin',1)
test_data['Sex'] = test_data['Sex'].map({'male':1,'female':0})
test_data['Embarked2'] = test_data['Embarked'].map({'S':1,'C':1,'Q':0})

test_data['Embarked'] = test_data['Embarked'].map({'S':0,'C':1,'Q':1})
print(test_data.isnull().sum())

print(test_data.shape)
test = test_data.iloc[:,1:test_data.shape[0]]

print(test.head())

print(train.head())

test = test.fillna(0)
results=clf.predict(test)
print(results)
df = pd.DataFrame(results)

df.index.name='PassengerId'

df.index+=892

df.columns=['Survived']

df.to_csv('results.csv', header=True)

print(df)