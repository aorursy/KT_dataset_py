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
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

train_data.Age
train_data.isnull().sum()
train_data.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis = 1 , inplace= True)
train_data.head()
sex = train_data.Sex
train_data.drop(['Sex'], axis = 1, inplace = True)
imputer = train_data.copy()

impute = SimpleImputer(strategy = 'mean')

train_data = pd.DataFrame(impute.fit_transform(train_data))

train_data.columns = imputer.columns

train_data.Age
train_data = pd.concat([train_data, sex], axis=1)
train_data.head()
gender = pd.get_dummies(train_data['Sex'])

train_data = pd.concat([train_data, gender], axis=1)

train_data.head()
train_data.drop(['Sex', 'female'], axis =1, inplace=True)

train_data.head()
test_data.head()
test_data1 = test_data.copy()

test_data1.head()
test_data.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis = 1 , inplace= True)
test_data.isnull().sum()
sex_test = test_data.Sex
test_data.drop(['Sex'], axis=1, inplace=True)
imputer_test = test_data.copy()

impute = SimpleImputer(strategy = 'mean')

test_data = pd.DataFrame(impute.fit_transform(test_data))

test_data.columns = imputer_test.columns

test_data.isnull().sum()
test_data.Age
test_data = pd.concat([test_data, sex_test], axis=1)
gender1 = pd.get_dummies(test_data['Sex'])

test_data = pd.concat([test_data, gender1], axis=1)

test_data.head()
test_data.drop(['Sex', 'female'], axis =1, inplace=True)

test_data.head()
y = train_data.Survived

train_data.drop(['Survived'], axis = 1, inplace =True)

X =train_data
model = RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state=1)

model.fit(X,y)
predicitons = model.predict(test_data)
output = pd.DataFrame({'PassengerId': test_data1.PassengerId, 'Survived': predicitons})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")