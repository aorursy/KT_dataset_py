# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression # machine learning model

from sklearn.ensemble import RandomForestClassifier # Imports Random forest model 



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
# Create filled age based on mean for train data

train_data["Age_filled"] = train_data["Age"].fillna(train_data["Age"].mean())

train_data.info()
# Create filled age based on mean for test data

test_data["Age_filled"] = test_data["Age"].fillna(test_data["Age"].mean())

test_data.info()
# Converts Sex to category for train and test and shows categorys

train_data["Sex"] = train_data["Sex"].astype("category")

test_data["Sex"] = test_data["Sex"].astype("category")

print(train_data["Sex"])

train_data.info()

print(test_data["Sex"])

test_data.info()
# Create surivived variable

y_train =  train_data["Survived"].copy()

print(y_train)
# select and Copy features for training dataset

X_train =  train_data[["Sex","Age_filled", "Pclass"]].copy()

print(X_train)

X_train.info()
# select and Copy features for test dataset

X_test =  test_data[["Sex","Age_filled", "Pclass"]].copy()

print(X_test)

X_test.info()
# replace categorical variables with its codes

X_train["Sex"] = X_train["Sex"].cat.codes

X_test["Sex"] = X_test["Sex"].cat.codes

X_train.info()

X_test.info()
# Creating model

clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

# fitting model to data

clf.fit(X_train, y_train)
# Saving predictions to varible 

predictions = clf.predict(X_test)

print(predictions)
# Writes to a file (for submission to Kaggle)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)
