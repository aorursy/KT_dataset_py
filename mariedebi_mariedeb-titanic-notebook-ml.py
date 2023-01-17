# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Import sklearn modules

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
gender_data = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

gender_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
##Define variable "Y" that needs to be accessed : survival

y = train_data["Survived"]



##Select features that will determine the survival of passenger

features = ["Pclass", "Sex", "SibSp", "Parch"]



##Create X in train and in test

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



##Create model

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



##Save output

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
pd.read_csv('my_submission.csv')
survivers = output["Survived"]

rate_survivers = sum(survivers) / len(survivers)



print("% of people who survived:", rate_survivers)
##Define variable "Y" that needs to be accessed : survival

y = train_data["Survived"]



##Select features that will determine the survival of passenger

features = ["Pclass", "SibSp", "Parch"]



##Create X 

X = train_data[features]



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Specify Model

titanic_model = RandomForestRegressor(random_state=1)

# Fit Model

titanic_model.fit(train_X, train_y)



# Make validation predictions and calculate mean absolute error

val_predictions = titanic_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(val_mae))



##Save output

output_forest1 = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': val_predictions})

output_forest1.to_csv('my_submission2.csv', index=False)

print("Your submission was successfully saved!")
