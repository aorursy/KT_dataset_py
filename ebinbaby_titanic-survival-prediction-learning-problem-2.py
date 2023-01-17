#Importing the libraries



import pandas as pd

import numpy as np

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#Reading the Training Data

train_data = pd.read_csv('../input/titanic/train.csv')

data.head()
#Reading the Test Data

test_data = pd.read_csv('../input/titanic/test.csv')

test_data.head()
#analyzing the Test data

women = train_data.loc[train_data.Sex == 'female']['Survived']

rate_women = sum(women)/len(women)

print("% women survived: ",rate_women)



men = train_data.loc[train_data.Sex == 'male']['Survived']

rate_men = sum(men)/len(men)

print("% Men survived: {}".format(rate_men))
#First Machine Learning Model

from sklearn.ensemble import RandomForestClassifier

y = train_data['Survived']

features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")