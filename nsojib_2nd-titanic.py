# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

train.head()
y=train['Survived']
X=train.drop(['PassengerId', 'Survived', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin'], axis=1)

X_test=test.drop(['PassengerId',  'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin'], axis=1)

X.head()

# X_test.head()
from sklearn.preprocessing import LabelEncoder

labelEncoder_X = LabelEncoder()

X.Sex=labelEncoder_X.fit_transform(X.Sex)

X_test.Sex=labelEncoder_X.fit_transform(X_test.Sex)

X.head()
# number of null values in embarked:

print ('Number of null values in Embarked:', sum(X.Embarked.isnull()))

print ('Number of null values in Name:', sum(X.Name.isnull()))

print ('Number of null values in Age:', sum(X.Age.isnull()))

print ('Number of null values in Sex:', sum(X.Sex.isnull()))
# number of null values in embarked:

print ('Number of null values in Embarked:', sum(X_test.Embarked.isnull()))

print ('Number of null values in Name:', sum(X_test.Name.isnull()))

print ('Number of null values in Age:', sum(X_test.Age.isnull()))

print ('Number of null values in Sex:', sum(X_test.Sex.isnull()))
# fill the two values with one of the options (S, C or Q)

row_index = X.Embarked.isnull()

X.loc[row_index,'Embarked']='S' 
#Encoding Embarked

labelEncoder_X = LabelEncoder()

X.Embarked=labelEncoder_X.fit_transform(X.Embarked)

X_test.Embarked=labelEncoder_X.fit_transform(X_test.Embarked)

X.head()
X.Age.mean()
row_index = X.Age.isnull()

X.loc[row_index,'Age']=X.Age.mean()

row_index = X_test.Age.isnull()

X_test.loc[row_index,'Age']=X.Age.mean()
X.head()
X_test.head()
name= X.Name.str.split(',').str[1]

X.iloc[:,1]=pd.DataFrame(got).Name.str.split('\s+').str[1]

# xpc=pd.DataFrame(name).Name.str.split('\s+').str[1]

X.head()
name= X_test.Name.str.split(',').str[1]

X_test.iloc[:,1]=pd.DataFrame(got).Name.str.split('\s+').str[1]

# xpc=pd.DataFrame(name).Name.str.split('\s+').str[1]

X_test.head()
#Encoding Name Title

labelEncoder_X = LabelEncoder()

X.Name=labelEncoder_X.fit_transform(X.Name)

X_test.Name=labelEncoder_X.fit_transform(X_test.Name)

X.head()
X_test.head()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score
clf = LogisticRegression()

# accuracies = cross_val_score(estimator = clf, X=X , y=y , cv = 10)

# print("Logistic Regression:\n Accuracy:", accuracies.mean(), "+/-", accuracies.std(),"\n")
p=clf.fit(X,y)
pred =clf.predict(X_test)

print(pred)
sub = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': pred})

sub.head()
sub = sub.to_csv('res.csv', index=False)
print(os.listdir("../"))