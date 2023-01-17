# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Import 'tree' from scikit-learn library

from sklearn import tree



# Import the `RandomForestClassifier`

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.


# Load the train and test datasets to create two DataFrames

#train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"

train = pd.read_csv("../input/train.csv")



#test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"

test = pd.read_csv("../input/test.csv")



#Print the `head` of the train and test dataframes

print(train.head())

print(test.head())



# Convert the male and female groups to integer form

train["Sex"][train["Sex"] == "male"] = 0

train["Sex"][train["Sex"] == "female"] = 1



test["Sex"][test["Sex"] == "male"] = 0

test["Sex"][test["Sex"] == "female"] = 1



# Impute the Embarked variable

train["Embarked"] = train["Embarked"].fillna("S")



# Convert the Embarked classes to integer form

train["Embarked"][train["Embarked"] == "S"] = 0

train["Embarked"][train["Embarked"] == "C"] = 1

train["Embarked"][train["Embarked"] == "Q"] = 2



test["Embarked"][test["Embarked"] == "S"] = 0

test["Embarked"][test["Embarked"] == "C"] = 1

test["Embarked"][test["Embarked"] == "Q"] = 2



#Clean up

train["Age"][np.isnan(train["Age"])] = train["Age"].median()

test["Age"][np.isnan(test["Age"])] = test["Age"].median()



#Print the Sex and Embarked columns

print(train["Embarked"])

print(train["Sex"])



# Print the train data to see the available features

print(train)



# Create the target and features numpy arrays: target, features_one

target = train["Survived"].values

features_one = train[["Pclass", "Sex", "Age", "Fare"]].values





# Fit your first decision tree: my_tree_one

my_tree_one = tree.DecisionTreeClassifier()

my_tree_one = my_tree_one.fit(features_one, target)



# Look at the importance and score of the included features

print(my_tree_one.feature_importances_)

print(my_tree_one.score(X=features_one, y=target))



# Impute the missing value with the median

test.Fare[152] = test["Fare"].median()





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

my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])

print(my_solution)



# Create a new array with the added features: features_two

print(train.describe())



features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch" , "Embarked"]].values



#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two

max_depth = 10

min_samples_split = 5

my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 20, random_state = 1)

my_tree_two = my_tree_two.fit(features_two, target)



#Print the score of the new decison tree

print(my_tree_two.score(features_two, target))



# Create train_two with the newly defined feature

train_two = train.copy()

train_two["family_size"] = train_two["SibSp"] + train_two["Parch"] + 1



# Create a new feature set and add the new feature

features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values



# Define the tree classifier, then fit the model

my_tree_three = tree.DecisionTreeClassifier()

my_tree_three = my_tree_three.fit(features_three, target)



# Print the score of this decision tree

print(my_tree_three.score(features_three, target))



# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables

features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values



# Building and fitting my_forest

forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)

my_forest = forest.fit(features_forest, target)



# Print the score of the fitted random forest

print(my_forest.score(features_forest, target))



# Compute predictions on our test set features then print the length of the prediction vector

test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

pred_forest = my_forest.predict(test_features)

print(len(pred_forest))



print(train.describe())

print(my_tree_two.feature_importances_)

print(my_forest.feature_importances_)



#Compute and print the mean accuracy score for both models

print(my_tree_two.score(features_two, target))

print(my_forest.score(features_forest, target))