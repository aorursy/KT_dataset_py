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
df1 = pd.read_csv("/kaggle/input/titanic/train.csv")

df2 = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

df3 = pd.read_csv("/kaggle/input/titanic/test.csv")
print(df1["Age"].isnull().sum())

print(df1["Sex"].isnull().sum())

print(df1["Pclass"].isnull().sum())

print(df1["SibSp"].isnull().sum())

print(df1["Parch"].isnull().sum())

print(df1["Fare"].isnull().sum())

print(df1["Embarked"].isnull().sum())

print(df1["Survived"].isnull().sum())

print(df1["PassengerId"].isnull().sum())
ohe1 = pd.get_dummies(df1['Sex'])

df1.drop(['Sex'],axis =1)

df1['Sex'] = ohe1

ohe2 = pd.get_dummies(df1['Embarked'])

df1.drop(['Embarked'],axis =1)

df1['Embarked'] = ohe1
df1.drop(["Name","Ticket","Cabin"],axis = 1,inplace = True)
df1["Age"].fillna(df1["Age"].mean(),inplace = True)
print(df1["Age"].isnull().sum())
df1.corr()
x_train = df1.iloc[ : ,2:]

y_train = df1.iloc[ : ,1:2]
print(df3["Age"].isnull().sum())

print(df3["Sex"].isnull().sum())

print(df3["Pclass"].isnull().sum())

print(df3["SibSp"].isnull().sum())

print(df3["Parch"].isnull().sum())

print(df3["Fare"].isnull().sum())

print(df3["Embarked"].isnull().sum())

print(df3["PassengerId"].isnull().sum())
ohe3 = pd.get_dummies(df3['Sex'])

df3.drop(['Sex'],axis =1)

df3['Sex'] = ohe3

ohe4 = pd.get_dummies(df3['Embarked'])

df3.drop(['Embarked'],axis =1)

df3['Embarked'] = ohe4

df3.drop(["Name","Ticket","Cabin"],axis = 1,inplace = True)
df3["Age"].fillna(df1["Age"].mean(),inplace = True)

df3["Fare"].fillna(df1["Fare"].mean(),inplace = True)
#import sklearn

from sklearn.ensemble import RandomForestClassifier
clf1 = RandomForestClassifier(n_estimators=100, max_depth=100)
clf1.fit(x_train,y_train)
op = df3.iloc[ : ,0:1]

df3.drop(["PassengerId"],inplace =True,axis =1)
pred1 = clf1.predict(df3)
op["Survived"] = pred1
op.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")