# Import the Pandas library

import pandas as pd

# Import the Numpy library

import numpy as np

# Import 'tree' from scikit-learn library

from sklearn import tree



# Load the train and test datasets to create two DataFrames

test = pd.read_csv("../input/test.csv")



#Print the `head` of the train and test dataframes

print(test.describe())



# Create the column Child and assign to 'NaN'

test["Child"] = float('NaN')



print(test["Child"])

# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.



test["Child"][test["Age"] < 18] = 1

test["Child"][test["Age"] >= 18] = 0



print(test["Child"])



# Initialize a Survived column to 0

test["Survived"] = 0



# Set Survived to 1 if Sex equals "female" and print the `Survived` column from `test_one`

test["Survived"][test["Sex"] == "female"] = 1

test["Survived"][test["Sex"] == "male"] = 0



print(test["Survived"])



# Print normalized Survival Rates for passengers under 18

#print(test["Survived"][test["Child"] == 1].value_counts(normalize = True))



# Print normalized Survival Rates for passengers 18 or older

#print(test["Survived"][test["Child"] == 0].value_counts(normalize = True))





# Create the target and features numpy arrays: target, features_one

target = test["Survived"].values

features_one = test[["Pclass", "Sex", "Age", "Fare"]].values



# Fit your first decision tree: my_tree_one

my_tree_one = tree.DecisionTreeClassifier()

my_tree_one = my_tree_one.fit( features_one,target)



# Look at the importance and score of the included features

print(my_tree_one.feature_importances_)

print(my_tree_one.score(features_one, target))



# Extract the features from the test set: Pclass, Sex, Age, and Fare.

test_features = test[["Pclass", "Sex", "Age", "Fare"]].values



# Make your prediction using the test set

my_prediction = my_tree_one.predict(test_features)



# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions

PassengerId =np.array(test["PassengerId"]).astype(int)

my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

print(my_solution)



# Check that your data frame has 418 entries

print(my_solution.shape)



# Write your solution to a csv file with the name my_solution.csv

my_solution.to_csv("my_solution.csv", index_label = ["PassengerId"])

sol = pd.read_csv("my_solution.csv")



print(sol)
