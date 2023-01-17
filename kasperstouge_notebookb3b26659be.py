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
import numpy as np

import pandas as pd
train_data = pd.read_csv("../input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("../input/titanic/test.csv")

test_data.head()
train_data.info()
test_data.info()
train_data["Embarked"].describe()

train_data["Embarked_filled"] = train_data["Embarked"].fillna("S")
test_data["Embarked"].describe()

test_data["Embarked_filled"] = test_data["Embarked"].fillna("S")
train_data["relatives"] = train_data["SibSp"] + train_data["Parch"]

test_data["relatives"] = test_data["SibSp"] + test_data["Parch"]
train_data["Sex_cat"] = train_data["Sex"].astype("category")
test_data["Sex_cat"] = test_data["Sex"].astype("category")
train_data["Embarked_filled"] = train_data["Embarked_filled"].astype("category")

test_data["Embarked_filled"] = test_data["Embarked_filled"].astype("category")
train_data["Fare_filled"] = train_data["Fare"].fillna(0)

test_data["Fare_filled"] = test_data["Fare"].fillna(0)
train_data["Fare_filled"] = train_data["Fare_filled"].astype(int)

test_data["Fare_filled"] = test_data["Fare_filled"].astype(int)
train_data["Cabin_filled"] = train_data["Cabin"].fillna("U0")



deck = []

for data in train_data["Cabin_filled"]:

    if data[0][0] == "U":

        deck.append("U")

    elif data[0][0] == "A":

        deck.append("A")

    elif data[0][0] == "B":

        deck.append("B")

    elif data[0][0] == "C":

        deck.append("C")

    elif data[0][0] == "D":

        deck.append("D")

    elif data[0][0] == "E":

        deck.append("E")

    elif data[0][0] == "F":

        deck.append("F")

    elif data[0][0] == "G":

        deck.append("G")

        

deck.append("U")



train_data["Deck"] = deck
test_data["Cabin_filled"] = test_data["Cabin"].fillna("U0")



deck_test = []

for data in test_data["Cabin_filled"]:

    if data[0][0] == "U":

        deck_test.append("U")

    elif data[0][0] == "A":

        deck_test.append("A")

    elif data[0][0] == "B":

        deck_test.append("B")

    elif data[0][0] == "C":

        deck_test.append("C")

    elif data[0][0] == "D":

        deck_test.append("D")

    elif data[0][0] == "E":

        deck_test.append("E")

    elif data[0][0] == "F":

        deck_test.append("F")

    elif data[0][0] == "G":

        deck_test.append("G")

        

test_data["Deck"] = deck_test
train_data["Deck_cat"] = train_data["Deck"].astype("category")

test_data["Deck_cat"] = test_data["Deck"].astype("category")
train_data.info()
from sklearn.linear_model import LogisticRegression
y = train_data["Survived"]
X = train_data[["Pclass", "Sex_cat", "relatives", "Deck_cat"]]

X_test = test_data[["Pclass", "Sex_cat", "relatives", "Deck_cat"]]
X["Sex_cat"] = X["Sex_cat"].cat.codes

X_test["Sex_cat"] = X_test["Sex_cat"].cat.codes
X["Deck_cat"] = X["Deck_cat"].cat.codes

X_test["Deck_cat"] = X_test["Deck_cat"].cat.codes
X.info()
X_test.info()
model = LogisticRegression()
model.fit(X, y)
predictions = model.predict(X_test)
predictions
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")