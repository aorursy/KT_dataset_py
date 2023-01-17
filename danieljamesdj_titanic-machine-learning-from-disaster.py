import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

gender_submission.head()


test = pd.read_csv("../input/titanic/test.csv")

test.head()
train = pd.read_csv("../input/titanic/train.csv")

train.head()
columnStats = pd.DataFrame()

columnStats["Column"] = train.columns.values

column_count = []

column_unique = []

column_nullCount = []

for column in train.columns.values:

    column_count.append(len(train[column]))

    column_unique.append(len(train[column].unique()))

    column_nullCount.append(train[column].isnull().sum())

columnStats["count"] = column_count

columnStats["unique"] = column_unique

columnStats["nullCount"] = column_nullCount

columnStats.head(12)
train = train.drop('PassengerId', axis=1).drop('Name', axis=1).drop('Ticket', axis=1).drop('Cabin', axis=1)

print(train)
train["Age"].fillna(train["Age"].mean(), inplace=True)

train["Embarked"].fillna("Unknown", inplace=True)
columns = train.columns.values

for column in np.delete(columns, np.argwhere(columns == "Survived")):

    _, axis = plt.subplots()

    for value in train[column].unique():

        axis.bar(value, len(train[train[column] == value][train["Survived"] == 1]), label=value)

    axis.set_title(column + ": Survived")

    plt.show()

    _, axis = plt.subplots()

    for value in train[column].unique():

        axis.bar(value, len(train[train[column] == value][train["Survived"] == 0]), label=value)

    axis.set_title(column + ": Didn't survive")

    plt.show()
from sklearn.naive_bayes import GaussianNB

from sklearn import preprocessing



train['isMale'] = train['Sex'] == 'male'

test['isMale'] = test['Sex'] == 'male'

x_input = []

x_test = []

for isMale in train['isMale']:

    x_input.append([isMale])

for isMale in test['isMale']:

    x_test.append([isMale])

x_input = np.array(x_input)

x_test = np.array(x_test)



gaussianNB = GaussianNB()

gaussianNB.fit(x_input, train["Survived"])



score = gaussianNB.score(x_test, gender_submission["Survived"])

result = gaussianNB.predict(x_test)

submission = pd.DataFrame({

    "PassengerId": test["PassengerId"],

    "Survived": gender_submission["Survived"]

})

submission.to_csv('my_submission.csv', index=False)