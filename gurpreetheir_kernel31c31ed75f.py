# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

import math

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
test = pd.read_csv("/kaggle/input/titanic/test.csv")

train = pd.read_csv("/kaggle/input/titanic/train.csv")

train.hist(bins=50,figsize=(20,15))
X,y = train.drop("Survived",axis=1),train["Survived"]

X.head()
def CheckNull(X):

    null_columns=X.columns[X.isnull().any()]

    print(X[null_columns].isnull().sum())

def convertToNum(X):

    total=0

    counter=0

    for i in range(len(X)):

        if(X[i]!="0"):

            sum = (ord(X[i][0])-64)*100 + int(X[i][1:])

            X[i]=sum

            total+=sum

            counter+=1

    avg = total/counter

    for i in range(len(X)):

        if(X[i]=="0"):

            X[i]=avg

    return X

            

    


X_att = X.drop("Name",axis=1)

X_att = X_att.drop("Ticket",axis=1)

X_att = X_att.drop("Cabin",axis=1)

median = X["Age"].median()

faremed = X["Fare"].median()

X_att["Age"].fillna(median, inplace=True)

X_att["Embarked"].fillna("S", inplace=True)



CheckNull(X_att)

X_att.loc[X_att["Sex"]=="male","Sex"]=1

X_att.loc[X_att["Sex"]=="female","Sex"]=0

X_att.loc[X_att["Embarked"]=="Q","Embarked"]=2

X_att.loc[X_att["Embarked"]=="S","Embarked"]=1

X_att.loc[X_att["Embarked"]=="C","Embarked"]=0

def CleanExample(x):

    x_att = x.drop("Name",axis=1)

    x_att = x_att.drop("Ticket",axis=1)

    x_att = x_att.drop("Cabin",axis=1)

    x_att["Age"].fillna(median, inplace=True)

    x_att["Fare"].fillna(faremed, inplace=True)

    x_att["Embarked"].fillna("S", inplace=True)

    x_att.loc[x_att["Sex"]=="male","Sex"]=1

    x_att.loc[x_att["Sex"]=="female","Sex"]=0

    x_att.loc[x_att["Embarked"]=="Q","Embarked"]=2

    x_att.loc[x_att["Embarked"]=="S","Embarked"]=1

    x_att.loc[x_att["Embarked"]=="C","Embarked"]=0

    return x_att

X_att.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

model = RandomForestClassifier(n_estimators=100,min_samples_leaf=2)

model_t = cross_val_score(model, X_att, y, cv=10)

model_t
from sklearn.metrics import accuracy_score

model.fit(X_att,y)

y_pred = model.predict(X_att)

accuracy_score(y, y_pred)
clean_test = CleanExample(test)

CheckNull(clean_test)

test_pred = model.predict(clean_test)

submission = clean_test["PassengerId"]



sub = pd.DataFrame(submission)

sub = sub.assign(Survived= test_pred.tolist()) 

sub.to_csv(r'submissiond.csv',index=False)

sub