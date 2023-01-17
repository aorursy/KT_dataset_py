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
import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")



train_data = train_df.drop(["Name", "PassengerId", "Cabin", "Ticket"], axis=1)

test_data = test_df.drop(["Name", "PassengerId", "Cabin", "Ticket"], axis=1)



train_data.head()
# Fill NaN values with mean of corresponding column

train_data["Age"] = train_data["Age"].fillna(train_data["Age"].mean())

test_data["Age"] = test_data["Age"].fillna(test_data["Age"].mean())

test_data["Fare"] = test_data["Fare"].fillna(test_data["Fare"].mean())



train_data["Sex"] = train_data["Sex"].apply(lambda x: 1 if (x == "female") else 0 )

test_data["Sex"] = test_data["Sex"].apply(lambda x: 1 if (x == "female") else 0 )



train_data.head()
# Convert dataframe to dummy variables

train_data = pd.get_dummies(train_data, columns=["Embarked"])

test_data = pd.get_dummies(test_data, columns=["Embarked"])
train_data.head()
X = train_data.drop("Survived", axis=1)

Y = train_data["Survived"]

X.head()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
model = RandomForestClassifier()

model.fit(x_train, y_train)
prediction = model.predict(test_data)
model.score(x_test, y_test)
result = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": prediction})
result.to_csv("survival_predictor.csv", index=False)