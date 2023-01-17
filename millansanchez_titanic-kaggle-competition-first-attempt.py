# Import the Pandas library

import pandas as pd



# Import the numpy library

import numpy as np



# Import 'tree' from scikit-learn library

from sklearn import tree
# Load the train and test datasets to create two DataFrames

train_url = "../input/train.csv"

train = pd.read_csv(train_url)





test_url = "../input//test.csv"

test = pd.read_csv(test_url)
#Print the `head` of the train and test dataframes

print(train.head())

print(test.head())
#Impute the "Age" feature with the mean

train["Age"]=train["Age"].fillna(train["Age"].mean())
# Convert the male and female groups to integer form

train["Sex"] = train["Sex"].map({'female': 1, 'male': 0})
# Create the target and features numpy arrays: target, features_one

target = train["Survived"].values

features = train[["Pclass", "Sex", "Age", "Fare"]].values
# Fit your first decision tree: my_tree

my_tree = tree.DecisionTreeClassifier()

my_tree = my_tree.fit(features, target)
# Look at the importance and score of the included features

print(my_tree.feature_importances_)

print(my_tree.score(features, target))
# Convert the male and female groups to integer form

test["Sex"] = test["Sex"].map({'female': 1, 'male': 0})



#Impute the "Age" feature with the mean

test["Age"]=test["Age"].fillna(test["Age"].mean())
#Impute the "Fare" feature with the mean

test["Fare"]=test["Fare"].fillna(test["Fare"].mean())
# Extract the features from the test set: Pclass, Sex, Age, and Fare.

test_features = test[["Pclass", "Sex", "Age", "Fare"]].values
# Make your prediction using the test set

my_prediction = my_tree.predict(test_features)
PassengerId =np.array(test["PassengerId"]).astype(int)

my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

print(my_solution)

# Write your solution to a csv file with the name my_solution.csv

my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])