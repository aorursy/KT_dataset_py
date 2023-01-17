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
print(train_data["Embarked"].describe())
from sklearn.ensemble import RandomForestClassifier



# Get our y (only for train - Kaggle doesn't give us the test target)

y_train = train_data["Survived"]



# transform Sex to categorical

train_data["Sex"] = train_data["Sex"].astype("category")

test_data["Sex"] = test_data["Sex"].astype("category")



# calculating the number of relatives for each passenger 

train_data["Relatives"] = train_data["SibSp"] + train_data["Parch"]

test_data["Relatives"] = test_data["SibSp"] + test_data["Parch"]



# Creating a new column 'Age_filled' where missing values are filled with the mean age.

MeanAge = train_data['Age'].mean()

MeanAge = int(MeanAge)

train_data["Age_filled"] = train_data["Age"].fillna(MeanAge)

test_data["Age_filled"] = test_data["Age"].fillna(MeanAge)



# Filling embarked with the most common value. Transforming Embarked to categorical, and renaming the categories to their codes.

most_common_embarked = "S"

train_data["Embarked"] = train_data["Embarked"].fillna(most_common_embarked)



train_data["Embarked"] = train_data["Embarked"].astype("category")

test_data["Embarked"] = test_data["Embarked"].astype("category")



train_data["Embarked_cat_codes"] = train_data["Embarked"].cat.codes

test_data["Embarked_cat_codes"] = test_data["Embarked"].cat.codes



# Get our X (train and test)

features = ["Pclass", "Sex", "Relatives", "Age_filled", "Embarked_cat_codes"]

X_test = test_data[features].copy()

X_train = train_data[features].copy()



# Change the Sex categorical to codes (train and test)

X_train["Sex"] = X_train["Sex"].cat.codes

X_test["Sex"] = X_test["Sex"].cat.codes



# Instantiate and fit a classifier

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X_train, y_train)
model.score(X_train, y_train)
# Get predictions

predictions = model.predict(X_test)



# Write to a file (for submission to Kaggle)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")