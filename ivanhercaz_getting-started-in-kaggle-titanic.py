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
# Load data sets

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

print(f"Number of rows: {len(train_data)}")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

print(f"Number of rows: {len(test_data)}")

test_data.head()
# Women who survived according to "train_data"

women_survived = train_data.loc[train_data.Sex == "female"]["Survived"]

rate_women = sum(women_survived)/len(women_survived) * 100



print(f"Women who survived: {rate_women} %")
# Male who survived according to "train_data"

male_survived = train_data.loc[train_data.Sex == "male"]["Survived"]

rate_male = sum(male_survived)/len(male_survived) * 100



print(f"Male who survived: {rate_male} %")
# Random Forest Model

from sklearn.ensemble import RandomForestClassifier



# The objective of our prediction

y = train_data["Survived"]



# Columns in which we're going to look for patterns

features = ["Pclass", "Sex", "SibSp", "Parch"]



# Get the dummies variables from the categorical data

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



# Create the model

model = RandomForestClassifier(

    n_estimators=100, # number of trees to create

    max_depth=5,

    random_state=1

)

# Fit the data to the model

model.fit(X, y)

# With the model and the data fitted we predict the features of the test data

predictions = model.predict(X_test)



output = pd.DataFrame({

    "PassengerId": test_data.PassengerId,

    "Survived": predictions

})

output.to_csv("getting_started_in_kaggle-titanic.csv", index=False)

print("Submission saved")