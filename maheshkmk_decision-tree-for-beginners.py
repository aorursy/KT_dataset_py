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
import pandas as pd

df = pd.read_csv('/kaggle/input/titanic/kaggle-titanic-master/kaggle-titanic-master/input/train.csv')

df.head()
m = df['Age'].median()

m
df['Age'] = df['Age'].fillna(m) 
isn = df['Age'].isnull()

isn
df1 = df.drop(['PassengerId','Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1)

df1.head()
from sklearn.preprocessing import LabelEncoder

sex_n = LabelEncoder()

df1['sex_n'] = sex_n.fit_transform(df1['Sex'])

df1.head()

df1 = df1.drop('Sex', axis=1)

df1.head()
from sklearn.model_selection import train_test_split
x = df1.drop('Survived', axis = 1)

y = df1['Survived']

# x.head()

# y.head()
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.15, random_state = 10)

# len(x_tran)

# len(x_test)
from sklearn import tree

model = tree.DecisionTreeClassifier()

model.fit(x_train, y_train)
model.predict(x_test)
model.score(x_test, y_test)