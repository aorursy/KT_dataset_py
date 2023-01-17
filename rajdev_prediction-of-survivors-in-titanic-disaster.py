import numpy as np

import pandas as pd

import sklearn as sk

from sklearn import tree



#Print you can execute arbitrary python code

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

print(train.head())

print(train.describe())

#Exploring the tarining data for survivals



# Passengers that survived vs passengers that passed away

print(train["Survived"].value_counts())



# As proportions

print(train["Survived"].value_counts(normalize = True))



# Males that survived vs males that passed away

print(train["Survived"][train["Sex"] == 'male'].value_counts())



# Females that survived vs Females that passed away

print(train["Survived"][train["Sex"] == 'female'].value_counts())



# Normalized male survival

print(train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True))



# Normalized female survival

print(train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True))

#Checking the age factor for survivals

# Create the column Child and assign to 'NaN'

train["Child"] = float('NaN')



# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.

train["Child"][train["Age"]<18]=1

train["Child"][train["Age"]>=18]=0

print(train["Child"].head())



# Print normalized Survival Rates for passengers under 18

print(train["Survived"][train["Child"] == 1].value_counts(normalize = True).head())



# Print normalized Survival Rates for passengers 18 or older

print(train["Survived"][train["Child"] == 0].value_counts(normalize = True).head())
#creating a first prediction. As the no. of survivals are 50% female and men had less chance to survive

#using this info will predict all females in the test set survive and all males in the test set die



# Create a copy of test: test_one



test_one=test

# Initialize a Survived column to 0

test_one["Survived"]=0



# Set Survived to 1 if Sex equals "female" and print the `Survived` column from `test_one`

test_one["Survived"][test_one["Sex"]== "female"] = 1

print(test_one["Survived"].head())

# Decision tree

import IPython

# Import 'tree' from scikit-learn library

from sklearn import tree



#cleaning the data before construction of trees

# Age has missing values, Sex and Embarked are categorical variables ; needs to covert this columns values into numerical values



#substitute each missing value with the median of the all present values.

train["Age"] = train["Age"].fillna(train["Age"].median())



# Convert the male and female groups to integer form

train["Sex"][train["Sex"] == "male"]=0

train["Sex"][train["Sex"]=="female"]=1



# Impute the Embarked variable

train["Embarked"] = train["Embarked"].fillna(train["Embarked"]=="S")



# Convert the Embarked classes to integer form

train["Embarked"][train["Embarked"] == "S"] = 0

train["Embarked"][train["Embarked"] == "C"] = 1

train["Embarked"][train["Embarked"] == "Q"] = 2



#Print the Sex and Embarked columns

print(train["Sex"].head())

print(train["Embarked"].head())







# Create the target and features numpy arrays: target (response), features_one

target = train["Survived"].values

features_one = train[["Pclass", "Sex", "Age", "Fare"]].values



# Fit your first decision tree: my_tree_one

my_tree_one = tree.DecisionTreeClassifier()

my_tree_one = my_tree_one.fit(features_one, target)



#to view the result of the decision tree the is to see the importance of the features that are included. 

#This is done by requesting the .feature_importances_ attribute of your tree object. 

#Another quick metric is the mean accuracy that you can compute using the .score() function

# Look at the importance and score of the included features



print(my_tree_one.feature_importances_)



#[ 0.12063997  0.31274009  0.23384062  0.33277932]



print(my_tree_one.score(features_one, target))



#0.977553310887 accuracy
#Decision tree



# Create a new array with the added features: features_two

features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values



#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two

max_depth = 10

min_samples_split = 5

my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)

my_tree_two = my_tree_two.fit(features_two, target)



#Print the score of the new decison tree, avg accuracy value

print(my_tree_two.score(features_two, target))

# Adding featuring engineering variable "family_size" to training data

# Create train_two with the newly defined feature

train_two = train.copy()

train_two["family_size"] = train["SibSp"] + train["Parch"] + 1



# Create a new feature set and add the new feature

features_three = train_two[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values



# Define the tree classifier, then fit the model

my_tree_three = tree.DecisionTreeClassifier()

my_tree_three = my_tree_three.fit(features_three, target)



# Print the score of this decision tree

print(my_tree_three.score(features_three, target))
#Building random forest 



# Import the `RandomForestClassifier`

from sklearn.ensemble import RandomForestClassifier



# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables

features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values



# Building and fitting my_forest

forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)

my_forest = forest.fit(features_forest, target)



# Print the score of the fitted random forest

print(my_forest.score(features_forest, target))

 # 0.939393939394  93% avg score of my_forest

    

# Compute predictions on our test set features then print the length of the prediction vector

test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

pred_forest = my_forest.predict(test_features)

print(len(pred_forest))

#418
#interpreting and comparing the tree and random forest model

#Request and print the `.feature_importances_` attribute

print(my_tree_two.feature_importances_)

#[ 0.14130255  0.17906027  0.41616727  0.17938711  0.05039699  0.01923751      0.0144483 ]



print(my_forest.feature_importances_)

#[ 0.10384741  0.20139027  0.31989322  0.24602858  0.05272693  0.04159232      0.03452128]



#On interpreting and comparing we can conclude that the most important feature was "Sex", but it was more significant for "my_tree_two"



#Compute and print the mean accuracy score for both models

print(my_tree_two.score(features_two, target))

#  0.905723905724 accuracy mean of the tree



print(my_forest.score(features_forest,target))

#  0.939393939394 accuracy mean of the forest