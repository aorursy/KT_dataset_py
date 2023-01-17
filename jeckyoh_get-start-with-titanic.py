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
train_data_path = '/kaggle/input/titanic/train.csv'

test_data_path = '/kaggle/input/titanic/test.csv'



train_data = pd.read_csv(train_data_path)

print("Successfully read train data.")

print(train_data.describe())



test_data = pd.read_csv(test_data_path)

print("Successfully read test data.")

print(test_data.describe())



print("Columns for train data:")

print(train_data.columns)
train_y = train_data['Survived']

features = ['Pclass', 'Sex', 'SibSp', 'Parch']

train_X = pd.get_dummies(train_data[features])



print(train_X.head())

print(train_y.head())
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, random_state= 0)

model = RandomForestClassifier(random_state = 1, max_depth = 5)

model.fit(train_X, train_y)

val_preds = model.predict(val_X)

print(mean_absolute_error(val_preds, val_y))
test_X = pd.get_dummies(test_data[features])

predictions = model.predict(test_X)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

print(output.head())
output.to_csv('submission.csv', index = False)