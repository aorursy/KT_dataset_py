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
# NaNs exist in column 'Age' in train_data and test_data

train_age_mean = round(train_data.Age.mean())

test_age_mean = round(test_data.Age.mean())

train_data.Age.fillna(train_age_mean, inplace=True)

test_data.Age.fillna(test_age_mean, inplace=True)



# NaNs exist in column 'Fare' in test_data

test_fare_mean = test_data.Fare.mean()

test_data.Fare.fillna(test_fare_mean, inplace=True)
train_y = train_data.Survived

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

train_X = pd.get_dummies(train_data[features])

test_X = pd.get_dummies(test_data[features])
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(train_X, train_y)

predictions = model.predict(test_X)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('titanic_submission.csv', index=False)

print('Congrats to myself, first project done!')