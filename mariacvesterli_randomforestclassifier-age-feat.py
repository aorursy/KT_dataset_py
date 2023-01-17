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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
train_data.info()
train_age_mean = train_data["Age"].mean()

train_age_filled = train_data.Age.fillna(train_age_mean)

train_data["Age"] = train_age_filled
embarked_filled = train_data.Embarked.fillna('S')

train_data["Embarked"] = embarked_filled

train_data.info()
test_data.info()
test_age_mean = test_data["Age"].mean()

test_age_filled = test_data.Age.fillna(test_age_mean)

test_data["Age"] = test_age_filled

test_data.info()
from sklearn.ensemble import RandomForestClassifier



# Get our y (only for train - Kaggle doesn't give us the test target)

y_train = train_data["Survived"]



# Transform Sex to categorical

train_data["Sex"] = train_data["Sex"].astype("category")

test_data["Sex"] = test_data["Sex"].astype("category")



#Transform Embarked to categorical

train_data["Embarked"] = train_data["Embarked"].astype("category")

test_data["Embarked"] = test_data["Embarked"].astype("category")



# Get our X (train and test)

features = ["Pclass", "Sex", "Age", "Embarked"]

X_test = test_data[features].copy()

X_train = train_data[features].copy()



# Change the Sex categorical to codes(train and test)

X_train["Sex"] = X_train["Sex"].cat.codes

X_test["Sex"] = X_test["Sex"].cat.codes



# Change the Embarked categorical to codes(train and test)

X_train["Embarked"] = X_train["Embarked"].cat.codes

X_test["Embarked"] = X_test["Embarked"].cat.codes



#Instantiate and fit a classifier

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X_train, y_train)



# Get predictions

predictions = model.predict(X_test)



# Write a file (for submission to Kaggle)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission is succesfully saved!")