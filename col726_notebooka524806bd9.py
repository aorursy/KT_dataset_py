import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import tree
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train["Sex"][train["Sex"] == "male"] = 0

train["Sex"][train["Sex"] == "female"] = 1
cabin_level_test = train["Cabin"].dropna()



first_letter = lambda x: x[0]



cabin_level_test = cabin_level_test.apply(first_letter)



print(cabin_level_test.value_counts())





train["Cabin"] = train["Cabin"].fillna("C").apply(first_letter)







print(train["Cabin"])
train["Cabin"][train["Cabin"] == "A"] = 0

train["Cabin"][train["Cabin"] == "B"] = 1

train["Cabin"][train["Cabin"] == "C"] = 2

train["Cabin"][train["Cabin"] == "D"] = 3

train["Cabin"][train["Cabin"] == "E"] = 4

train["Cabin"][train["Cabin"] == "F"] = 5

train["Cabin"][train["Cabin"] == "G"] = 6

train["Cabin"][train["Cabin"] == "T"] = 7



print(train["Cabin"].head())
target = train["Survived"].values

features_one = train[["Sex", "Fare", "Cabin"]].values



# Fit your first decision tree: my_tree_one

my_tree_one = tree.DecisionTreeClassifier()

my_tree_one = my_tree_one.fit(features_one, target)



# Look at the importance and score of the included features

print(my_tree_one.feature_importances_)

print(my_tree_one.score(features_one, target))
test["Sex"][test["Sex"] == "male"] = 0

test["Sex"][test["Sex"] == "female"] = 1



test["Cabin"] = test["Cabin"].fillna("C").apply(first_letter)



test["Cabin"][test["Cabin"] == "A"] = 0

test["Cabin"][test["Cabin"] == "B"] = 1

test["Cabin"][test["Cabin"] == "C"] = 2

test["Cabin"][test["Cabin"] == "D"] = 3

test["Cabin"][test["Cabin"] == "E"] = 4

test["Cabin"][test["Cabin"] == "F"] = 5

test["Cabin"][test["Cabin"] == "G"] = 6

test["Cabin"][test["Cabin"] == "T"] = 7



test.Fare[152] = test["Fare"].median()
test_features = test[["Sex", "Fare", "Cabin"]].values

my_prediction = my_tree_one.predict(test_features)



# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions

PassengerId =np.array(test["PassengerId"]).astype(int)

my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

print(my_solution)



# Check that your data frame has 418 entries

print(my_solution.shape)
my_solution.to_csv("my_solution.csv", index_label = ["PassengerId"])