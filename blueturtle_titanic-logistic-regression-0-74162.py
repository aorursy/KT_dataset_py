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

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
from sklearn.linear_model import LogisticRegression

from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score



y = train_data["Survived"] #Set y to the survived column from the training data

features = ["Pclass", "SibSp", "Parch", "Fare", "Age", "Sex"]



X_train = train_data[features]

X_test = test_data[features]



X_train = pd.get_dummies(train_data[features]) #Change all object types in features to dummy variables

X_test = pd.get_dummies(test_data[features]) #Change all object types in features to dummy variables



impute = SimpleImputer(strategy = "most_frequent")

#Find columns with empty values

X_train = pd.DataFrame(impute.fit_transform(X_train.values))

X_test = pd.DataFrame(impute.transform(X_test.values))





#Create model

model = LogisticRegression(max_iter = 1000, n_jobs=-1)

model.fit(X_train, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")

"""

acc_logreg = round(accuracy_score(predictions, y_test) * 100, 2)

print(acc_logreg)"""