# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
ds = pd.read_csv('/kaggle/input/titanic/train.csv')
ds.columns
ds.loc[:,['Survived', 'Pclass']]
ds.head()
ds.isna().sum()
df = ds.drop(['Cabin', 'PassengerId', 'Name', 'Ticket', 'Embarked'], axis=1)
df.columns
df.head()
df.shape
df.isna().sum()
df = df.dropna(axis=0)
df.shape
df.head()
genderMap = {
    'male' : 0,
    'female' : 1
}
df.Sex = df.Sex.map(genderMap)
df.head()
data = df.values
type(data)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data[:, 1:], data[:, 0])
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver= 'liblinear', multi_class= 'ovr')
lr.fit(X_train, y_train)
lr.score(X_test, y_test)
print(lr.coef_)
print(lr.intercept_)
df.keys()
df.describe()
m = (512.329200 - 0) /2
m
fares_normal = 2 * (df['Fare'] - m) / (512.329200-0)
fares_normal.min()
from sklearn.neighbors import KNeighborsClassifier
cl = KNeighborsClassifier()
cl.fit(X_train, y_train)
print(cl.score(X_train, y_train))
print(cl.score(X_test, y_test))
from sklearn.tree import DecisionTreeClassifier
cl = DecisionTreeClassifier()
cl.fit(X_train, y_train)
print(cl.score(X_train, y_train))
print(cl.score(X_test, y_test))


df['Fare'] = (df['Fare'] - 34.694514) / (512.329200 - 0)
fares_normal = (df['Fare'] - (512.329200/2)) / (512.329200)
fares_normal.mean()
df['Fare']
