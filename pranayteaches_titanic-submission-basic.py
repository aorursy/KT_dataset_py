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
train = pd.read_csv("/kaggle/input/titanic/train.csv")

train.head()
test = pd.read_csv("/kaggle/input/titanic/test.csv")

test_raw = test.copy()

test.head()
train.info()
train = train.drop(["Cabin", "PassengerId", "Name", "Age", "Ticket", "Fare"],axis=1)

train.head()
# viewing the data info

train.info()
#replacing null values with the mean of most appearing values in Embarked

train["Embarked"].fillna(train.Embarked.value_counts().idxmax(), inplace = True)

train.Embarked.value_counts()
# final view of the info of the data

train.info()
test.info()
test = test.drop(["Cabin", "PassengerId", "Name", "Age", "Ticket", "Fare"],axis=1)

test.head()
# viewing data

test.info()
y = train.Survived

X = pd.get_dummies(train.iloc[:,1:])

X_test = pd.get_dummies(test)

print(y.head())

print(X.head())

print(X_test.head())
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

y_pred = model.predict(X_test)

y_pred
sub = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

y_test = sub.Survived
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
output = pd.DataFrame({'PassengerId': test_raw.PassengerId, 'Survived': y_pred})

output.to_csv('my_submission.csv', index=False)

print("Submission successfully saved!")