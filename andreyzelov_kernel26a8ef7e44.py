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

import seaborn as sns

from pandas.plotting import scatter_matrix as sm

from matplotlib import pyplot as  plt

%matplotlib inline
data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
data.head()
data.isnull().sum()
data['Age'].median()

data.Age = data.Age.fillna(data['Age'].median())

data.Embarked = data.Embarked.fillna('S')



test_data['Age'].median()

test_data.Age = test_data.Age.fillna(test_data['Age'].median())

test_data.Embarked = test_data.Embarked.fillna('S')

test_data.Fare = test_data.Fare.fillna(test_data['Fare'].median())
data.describe()
target = data['Survived']

inputs = data.drop('Survived', axis='columns')

test_inputs = test_data
from sklearn.preprocessing import LabelEncoder as LE

le_sex = LE()

le_pclass = LE()

le_embarked = LE()
inputs['family'] = data['SibSp'] + data['Parch']

inputs['sex_n'] = le_sex.fit_transform(inputs['Sex'])

inputs['pclass_n'] = le_pclass.fit_transform(inputs['Pclass'])

inputs['Embarked'] = le_embarked.fit_transform(inputs['Embarked'].astype(str))



test_inputs['family'] = test_data['SibSp'] + test_data['Parch']

test_inputs['sex_n'] = le_sex.fit_transform(test_inputs['Sex'])

test_inputs['pclass_n'] = le_pclass.fit_transform(test_inputs['Pclass'])

test_inputs['Embarked'] = le_embarked.fit_transform(test_inputs['Embarked'].astype(str))
inputs.head()
inputs_n = inputs.drop(['PassengerId','pclass_n','Sex','SibSp','Parch','Name','Ticket','Cabin'], axis = 'columns')

inputs_t = inputs_n

inputs_t['Fare'] = inputs_t['Fare'].astype(int)



test_inputs_n = test_inputs.drop(['PassengerId','pclass_n','Sex','SibSp','Parch','Name','Ticket','Cabin'], axis = 'columns')

test_inputs_t = test_inputs_n

test_inputs_t['Fare'] = test_inputs_t['Fare'].astype(int)
from sklearn import tree

model = tree.DecisionTreeClassifier()

model.fit(inputs_t, target)
model.score(inputs_n, target)
ans = model.predict(test_inputs_t)
ans
test_data['Survived'] = ans

test_data[['PassengerId','Survived']].to_csv('gp_submit.csv', index=False)