# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# get the training data

train = pd.read_csv("/kaggle/input/titanic/train.csv")

train.head()
# get the test data

test = pd.read_csv("/kaggle/input/titanic/test.csv")

test.head()
# get training ground truth 

y = train["Survived"]

y
# get features (independent variables)

features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train[features])

X_test = pd.get_dummies(test[features])
# infer missing values

# from sklearn.impute import SimpleImputer

# my_imputer = SimpleImputer()

# X = my_imputer.fit_transform(X)

# X_test = my_imputer.fit_transform(X_test)
# train and fit a model

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)
model
# make predictions (y hat) on the test data

predictions = model.predict(X_test)

predictions 
# compile the predictions

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

output
# save to CSV

output.to_csv('my_submission.csv', index=False)