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

import numpy as np

df = pd.read_csv('/kaggle/input/titanicdtree/titanic.csv')

df.head()
inputs = df.drop(['PassengerId','Survived','Name','SibSp','Parch','Ticket','Cabin','Embarked'], axis = 'columns')

inputs.head()
target = df['Survived']

target.head()
inputs.Sex = inputs.Sex.map({'male':1,'female':2})
inputs.head()
inputs.info()
inputs.Age = inputs.Age.fillna(inputs.Age.mean())
inputs.Age[:10]
inputs.head()
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(inputs, target, test_size = 0.2)
print(len(X_train))

print(len(X_test))
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train,Y_train)
model.predict(X_test)
model.score(X_test,Y_test)