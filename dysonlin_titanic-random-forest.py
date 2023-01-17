import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier



# Load the train and test datasets to create two DataFrames

train_csv = "../input/train.csv"

train = pd.read_csv(train_csv)



test_csv = "../input/test.csv"

test = pd.read_csv(test_csv)



target = train["Survived"].values
# Clean data



# Train

# Impute the Age variable

train["Age"] = train["Age"].fillna(train["Age"].median())



# Convert the male and female groups to integer form

train["Sex"][train["Sex"] == "male"] = 0

train["Sex"][train["Sex"] == "female"] = 1



# Impute the Embarked variable

train["Embarked"] = train["Embarked"].fillna("S")



# Convert the Embarked classes to integer form

train["Embarked"][train["Embarked"] == "S"] = 0

train["Embarked"][train["Embarked"] == "C"] = 1

train["Embarked"][train["Embarked"] == "Q"] = 2



# Test

# Impute the Age variable

test["Age"] = test["Age"].fillna(test["Age"].median())



# Convert the male and female groups to integer form

test["Sex"][test["Sex"] == "male"] = 0

test["Sex"][test["Sex"] == "female"] = 1



# Impute the Embarked variable

test["Embarked"] = test["Embarked"].fillna("S")



# Convert the Embarked classes to integer form

test["Embarked"][test["Embarked"] == "S"] = 0

test["Embarked"][test["Embarked"] == "C"] = 1

test["Embarked"][test["Embarked"] == "Q"] = 2



# Impute the missing value with the median

test.Fare[152] = test.Fare.median()
#features = ["Sex", "Fare", "Age", "Pclass", "SibSp", "Parch", "Embarked"]

features = ["Sex", "Fare", "Age", "Pclass", "SibSp", "Parch", "Embarked"]

features_forest = train[features].values



# Building and fitting my_forest

forest = RandomForestClassifier(max_depth = 15, min_samples_split=2, 

                                n_estimators = 1000, random_state = 1)

my_forest = forest.fit(features_forest, target)



# Print the score of the fitted random forest

print(my_forest.score(features_forest, target))
# Compute predictions on our test set features then print the length of the prediction vector

test_features = test[features].values

pred_forest = my_forest.predict(test_features)

print(len(pred_forest))
# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions

PassengerId =np.array(test["PassengerId"]).astype(int)

my_solution = pd.DataFrame(pred_forest, PassengerId, columns = ["Survived"])

#print(my_solution)



# Check that your data frame has 418 entries

print(my_solution.shape)



# Write your solution to a csv file with the name my_solution.csv

my_solution.to_csv("my_forest.csv", index_label = ["PassengerId"])
#Request and print the `.feature_importances_` attribute

print(my_forest.feature_importances_)



#Compute and print the mean accuracy score

print(my_forest.score(features_forest, target))