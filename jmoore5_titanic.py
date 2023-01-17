import pandas as pd

import numpy as np

import sklearn as sk

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



train["Age"] = train["Age"].fillna(train.Age.median())

train.loc[train["Sex"] == "male", "Sex"] = 0

train.loc[train["Sex"] == "female", "Sex"] = 1



train["Embarked"] = train["Embarked"].fillna(train.Embarked.mode()[0])

train.loc[train["Embarked"] == "S", "Embarked"] = 0

train.loc[train["Embarked"] == "C", "Embarked"] = 1

train.loc[train["Embarked"] == "Q", "Embarked"] = 2



print(train.columns)

print(train.shape)
target = train["Survived"].values

features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values



forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)

my_forest = forest.fit(features_forest, target)



print(my_forest.score(features_forest, target))

test["Age"] = test["Age"].fillna(train.Age.median())

test.loc[test["Sex"] == "male", "Sex"] = 0

test.loc[test["Sex"] == "female", "Sex"] = 1



test["Embarked"] = test["Embarked"].fillna(train.Embarked.mode()[0])

test.loc[test["Embarked"] == "S", "Embarked"] = 0

test.loc[test["Embarked"] == "C", "Embarked"] = 1

test.loc[test["Embarked"] == "Q", "Embarked"] = 2



test["Fare"] = test["Fare"].fillna(train.Fare.median())



print(test.columns)

print(test.shape)





test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

pred_forest = my_forest.predict(test_features)



submission = pd.DataFrame(test["PassengerId"])

submission["Survived"] = pred_forest

submission.to_csv("kaggle.csv", index=False)