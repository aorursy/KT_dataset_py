# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/train.csv')

df.head()
df.fillna(0, inplace=True)

df.isnull().sum()
# split to x and y (features and target labels)

x = df.drop('Survived', axis=1)

y = df.Survived
x.head()
y.head()
x.PassengerId.nunique() == x.PassengerId.count()   # check that no dups of PassengerId
y.count() == df.shape[0] # check that x and y are correct
x.drop('PassengerId', axis=1, inplace=True)   # actually PassengerId should be useless

x = x.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)   # drop string columns; for first time skip sex (do not use LabelEncoding or OneHotEncoding)

x.head()
import sklearn

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score
clf = sklearn.tree.DecisionTreeClassifier(criterion='gini', max_depth=5,)
cross_val_score(clf, x, y, cv=10).mean()
clf.fit(x, y)
df2 = pd.read_csv('../input/test.csv')

df2.head()
t_id = df2.PassengerId

t = df2.drop(['PassengerId','Name','Sex','Ticket','Cabin','Embarked'], axis=1)

t.head()
t.fillna(0, inplace=True)

t.isnull().sum()
t.head()
p = clf.predict(t)
r = pd.DataFrame({'PassengerId': t_id,

             'Survived': p})

r.head()
r.to_csv('submit.csv', index=False)