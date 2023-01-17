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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head(100)

train_data['Fare'].describe()

train_data.shape

train_data['cab']=train_data['Cabin'].astype(str).str[0]

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()

test_data['cab']=test_data['Cabin'].astype(str).str[0]

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import Imputer

from sklearn.tree import DecisionTreeClassifier

from scipy import stats

import statistics



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])

X["Fare"]=train_data["Fare"].fillna(np.median(train_data["Fare"]))

X_test["Fare"]=test_data["Fare"].fillna(np.median(train_data["Fare"]))

#X["age"]=train_data["Age"].fillna(np.mean(train_data["Age"]))

#X_test["age"]=test_data["Age"].fillna(np.mean(test_data["Age"]))

X.head()

X_test.head()

#X_test.shape

model = LogisticRegression()

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
