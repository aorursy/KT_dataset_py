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
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
df1 = train[['PassengerId','Survived','Pclass','Sex','Age','SibSp','Parch','Fare']]
df1.head()
df1.shape
p = df1.isnull().sum().sum()

print(p)
df1_dl=df1.dropna(axis = 0)
q = df1_dl.isnull().sum().sum()

print(q)
df1_dl['Sex_value'] = df1_dl['Sex']
df1_dl
df1_dl.loc[df1_dl['Sex'] == 'male','Sex_value'] = 1

df1_dl.loc[df1_dl['Sex'] == 'female','Sex_value'] = 0
df1_dl
df1_train = df1_dl[['Survived','Pclass','Age','SibSp','Parch','Fare','Sex_value']]
df1_train
x_train = df1_train.iloc[:,1:]

y_train = df1_train.iloc[:,0]
x_train.head()
y_train.head()
test = pd.read_csv('/kaggle/input/titanic/test.csv')
test
df2 = test[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare']]

G_S = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
df2
G_S
df2['Survived'] = G_S['Survived']
df2
m = df2.isnull().sum().sum()

print(m)
df2_dl=df2.dropna(axis = 0)
n = df2_dl.isnull().sum().sum()

print(n)
df2_dl
df2_dl['Sex_value'] = df2_dl['Sex']

df2_dl.loc[df2_dl['Sex'] == 'male','Sex_value'] = 1

df2_dl.loc[df2_dl['Sex'] == 'female','Sex_value'] = 0
df2_dl
df2_test = df2_dl[['Pclass','Age','SibSp','Parch','Fare','Sex_value','Survived']]
df2_test
x_test = df2_test.iloc[:,0:6]

y_test = df2_test.iloc[:,6]
x_test
y_test
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
x1 = x_train.isnull().sum().sum()

y1 = y_train.isnull().sum().sum()

x2 = x_test.isnull().sum().sum()

y2 = y_test.isnull().sum().sum()

print(x1,y1,x2,y2)
from sklearn.linear_model import LogisticRegression



clf = LogisticRegression()

clf.fit(x_train,y_train)
clf.score(x_test,y_test)
clf.predict(x_test[301:306])
y_test[301:306]
from sklearn import svm



clf = svm.SVC()

clf.fit(x_train,y_train)
clf.score(x_test,y_test)
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier()

clf.fit(x_train,y_train)
clf.score(x_test,y_test)