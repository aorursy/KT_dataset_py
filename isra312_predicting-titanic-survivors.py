# Important Imports

import numpy as np

import pandas as pd

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from subprocess import check_output



# Input data files from the "../input/" directory.

print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



# Convert the male and female groups to integer form

train["Sex"][train["Sex"] == "male"] = 0

train["Sex"][train["Sex"] == "female"] = 1



test["Sex"][test["Sex"] == "male"] = 0

test["Sex"][test["Sex"] == "female"] = 1



# Imputing the age median for missing values 

train["Age"] = train["Age"].fillna(train["Age"].median())

test["Age"] = test["Age"].fillna(test["Age"].median())



# Impute the Embarked variable

train["Embarked"] = train["Embarked"].fillna("S")

test["Embarked"] = test["Embarked"].fillna("S")



# Convert the Embarked classes to integer form

train["Embarked"][train["Embarked"] == "S"] = 0

train["Embarked"][train["Embarked"] == "C"] = 1

train["Embarked"][train["Embarked"] == "Q"] = 2



test["Embarked"][test["Embarked"] == "S"] = 0

test["Embarked"][test["Embarked"] == "C"] = 1

test["Embarked"][test["Embarked"] == "Q"] = 2



# Impute the missing value with the median #test.Fare[152] = np.median(test["Fare"])

test.Fare[152] = test.Fare.median()



# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables

features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

target = train["Survived"].values



# Building and fitting my_forest

forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)

my_forest = forest.fit(features_forest, target)



# Print the score of the fitted random forest

print(my_forest.score(features_forest, target))



# Compute predictions on our test set features then print the length of the prediction vector

test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

pred_forest = my_forest.predict(test_features)

print(len(pred_forest))



# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions

PassengerId =np.array(test["PassengerId"]).astype(int)

my_solution = pd.DataFrame(pred_forest, PassengerId, columns = ["Survived"])



my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])