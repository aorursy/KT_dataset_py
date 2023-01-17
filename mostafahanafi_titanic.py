# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
train_data.head()
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
test_data.head()
women = train_data[train_data.Sex == 'female']['Survived']
women_survival_rate = sum(women)/len(women)*100

men = train_data[train_data.Sex == 'male']['Survived']
men_survival_rate = sum(men)/len(men)*100

print('{:.1f}% of women survived'.format(women_survival_rate))
print('{:.1f}% of men survived'.format(men_survival_rate))
train_data.Age.describe()
for i in range(0, 80, 10):
    age_group = train_data.loc[(train_data.Age > i) & (train_data.Age <= i+10)]
    survival = pd.Series( age_group.Survived )
    survival_rate = sum(survival)/len(survival)*100
    print('{:.1f}% of passengers aged {} to {} survived'.format(survival_rate, i, i+10))
train_data.count()
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor)
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

filtered_train_data = train_data.loc[ train_data.Age > 0 ]
missing_data = train_data.loc[ ~(train_data.Age > 0) ]

"""Define the model"""
age_model = RandomForestRegressor(random_state=1)

"""Defining variables for model fit"""
train_y = filtered_train_data.Age
variables=['Pclass', 'Sex', 'SibSp', 'Parch']
train_X = pd.get_dummies(filtered_train_data[variables])

"""Fit model"""
age_model.fit(train_X, train_y)

"""Predict missing ages"""
predicted_ages = age_model.predict( pd.get_dummies(missing_data[variables]) )

"""Replace them back into train data file"""
missing_data.loc[:, 'Age'] = predicted_ages
new_train_data = pd.concat([filtered_train_data, missing_data]).sort_values(by='PassengerId')
"""Predict survival of test data"""
