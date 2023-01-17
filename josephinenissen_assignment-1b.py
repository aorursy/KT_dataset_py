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
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
avr_age = train_data['Age'].mean()

train_data['Age'] = train_data['Age'].fillna(avr_age)

test_data['Age'] = test_data['Age'].fillna(avr_age)

train_data['Embarked'].describe()
common_embarked = 'S'

train_data['Embarked'] = train_data['Embarked'].fillna(common_embarked)
train_data['number_of_relatives'] = train_data['SibSp'] + train_data['Parch']

test_data['number_of_relatives'] = test_data['SibSp'] + test_data['Parch']
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier



# Get our y (only for train - Kaggle doesn't give us the test target)

y_train = train_data["Survived"]



# transform Sex to categorical

train_data["Sex"] = train_data["Sex"].astype("category")

test_data["Sex"] = test_data["Sex"].astype("category")



# Get our X (train and test)

features = ["Pclass", "Sex", "number_of_relatives", "Age"]

X_test = test_data[features].copy()

X_train = train_data[features].copy()



# Change the Sex categorical to codes (train and test)

X_train["Sex"] = X_train["Sex"].cat.codes

X_test["Sex"] = X_test["Sex"].cat.codes



# Instantiate and fit a classifier

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X_train, y_train)



# Get predictions

predictions = model.predict(X_test)



# Write to a file (for submission to Kaggle)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")

score = model.score(X_train, y_train)

print(score)