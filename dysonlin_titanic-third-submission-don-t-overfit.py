import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import tree 

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Load the train and test datasets to create two DataFrames

train_csv = "../input/train.csv"

train = pd.read_csv(train_csv)



test_csv = "../input/test.csv"

test = pd.read_csv(test_csv)



#Print the `head` of the train and test dataframes

print(train.head())

print(test.head())
# Passengers that survived vs passengers that passed away

print("Survived passengers:")

print(train["Survived"].value_counts())

# As proportions

print(train["Survived"].value_counts(normalize = True))

print(" ")



# Males that survived vs males that passed away

print("Survived male passengers:")

print(train["Survived"][train["Sex"] == 'male'].value_counts())

# Normalized male survival

print(train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True))

print(" ")



# Females that survived vs Females that passed away

print("Survived female passengers:")

print(train["Survived"][train["Sex"] == 'female'].value_counts())

# Normalized female survival

print(train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True))

print(" ")
# Create the column Child and assign to 0

train["Child"] = 0



# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.

train["Child"][train["Age"] < 18] = 1

train["Child"][train["Age"] >= 18] = 0

print(train["Child"].head())

print(" ")



# Print normalized Survival Rates for passengers under 18

print("Survived child passengers:")

print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))

print(" ")



# Print normalized Survival Rates for passengers 18 or older

print("Survived adult passengers:")

print(train["Survived"][train["Child"] == 0].value_counts(normalize = True))

print(" ")
# Create a copy of test: test_one

test_one = test[:]



# Initialize a Survived column to 0

test_one['Survived'] = 0



# Set Survived to 1 if Sex equals "female" and print the `Survived` column from `test_one`

test_one['Survived'][test_one['Sex'] == "female"] = 1

print(test_one['Survived'].head())
prediction = test_one[['PassengerId', 'Survived']]

print('Titanic: Second Submission: Decision Trees')

print(prediction.head())
prediction.to_csv('FemalesSurvive.csv', index=False)
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



#Print the Sex and Embarked columns

print(train["Sex"])

print(train["Embarked"])



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

print(my_tree_one.score(features_one, target))
# Print the train data to see the available features

print(train)



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



#Print 

print(test)



# Create the target and features numpy arrays: target, features_one

target = train["Survived"].values

features_one = train[["Pclass", "Sex", "Age", "Fare"]].values



# Fit your first decision tree: my_tree_one

my_tree_one = tree.DecisionTreeClassifier()

my_tree_one = my_tree_one.fit(features_one, target)



# Look at the importance and score of the included features

print(my_tree_one.feature_importances_)

print(my_tree_one.score(features_one, target))
# Impute the missing value with the median

test.Fare[152] = test.Fare.median()



# Extract the features from the test set: Pclass, Sex, Age, and Fare.

test_features = test[["Pclass", "Sex", "Age", "Fare"]].values



# Make your prediction using the test set

my_prediction = my_tree_one.predict(test_features)

print(my_prediction)



# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions

PassengerId =np.array(test["PassengerId"]).astype(int)

my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

print(my_solution)



# Check that your data frame has 418 entries

print(my_solution.shape)



# Write your solution to a csv file with the name my_solution.csv

my_solution.to_csv("my_tree_one.csv", index_label = ["PassengerId"])
# Create a new array with the added features: features_two

features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values



#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two

max_depth = 10

min_samples_split = 5

my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)



my_tree_two = my_tree_two.fit(features_two, target)



#Print the score of the new decison tree

print(my_tree_two.score(features_two, target))
# Extract the features from the test set: Pclass, Sex, Age, and Fare.

test_features = test[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values



# Make your prediction using the test set

my_prediction = my_tree_two.predict(test_features)

print(my_prediction)



# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions

PassengerId =np.array(test["PassengerId"]).astype(int)

my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

print(my_solution)



# Check that your data frame has 418 entries

print(my_solution.shape)



# Write your solution to a csv file with the name my_solution.csv

my_solution.to_csv("my_tree_two.csv", index_label = ["PassengerId"])