import pandas as pd



import seaborn as sns



import matplotlib.pyplot as plt



import numpy as np



from sklearn import tree



from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split







data = pd.read_csv("../input/titanic/train.csv")

data1 = pd.read_csv("../input/titanic/test.csv")

data1.head()
data.head()
data.tail()
data.dtypes
data.columns
data.iloc[45]
data.iloc[60]
data.iloc[400]
data.shape
data.describe
data.iloc[0:8].corr()
data['Age'].fillna(data['Age'].mean())
data.iloc[0:80].corr()
data.iloc[0:405].corr()
data.iloc[1:5].corr()
data.iloc[2:5].corr()
data.iloc[1:110].corr()
data.describe()
data1.count()
data1.describe()
data.columns
data1.isnull().sum()
data.isnull().sum()
data.columns
data.drop(['Name','Sex','Ticket','Cabin','Embarked','Pclass'],axis='columns',inplace=True)
data.shape
data.columns
data.dtypes
data.describe
data1.columns
# Passengers that survived vs passengers that passed away

print(data.Survived.value_counts())

# As proportions

print(data["Survived"].value_counts(normalize = True))

# Create the column Child and assign to 'NaN'

data["Child"] = float('NaN')

# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.

data["Child"][data["Age"] < 18] = 1

data["Child"][data["Age"] >= 18] = 0

print(data["Child"])

# Print normalized Survival Rates for passengers under 18

print(data["Survived"][data["Child"] == 1].value_counts(normalize = True))

# Print normalized Survival Rates for passengers 18 or older

print(data["Survived"][data["Child"] == 0].value_counts(normalize = True))

# Create a copy of test: data1

test = data1.copy()

# Initialize a Survived column to 0

test["Survived"] = 0
# Set Survived to 1 if Sex equals "female"

test["Survived"][test["Sex"] == "female"] = 1

print(test.Survived)
# Impute the Embarked variable

test["Embarked"] = test["Embarked"].fillna("S")

# Convert the Embarked classes to integer form

test["Embarked"][test["Embarked"] == "S"] = 0

test["Embarked"][test["Embarked"] == "C"] = 1

test["Embarked"][test["Embarked"] == "Q"] = 2

# Print the Sex and Embarked columns

print(test["Sex"])

print(test["Embarked"])
# Print the  data to see the available features

print(data)
data.columns
# Create the target and features numpy arrays: target, features_one

target = data["Survived"].values

features_one = data[["PassengerId","SibSp" , "Fare"]].values

# Print the train data to see the available features

print(data)



# Create the target and features numpy arrays: target, features_one

target = data["Survived"].values

features_one = data[["PassengerId", "Fare"]].values



# Fit your first decision tree: my_tree_one

my_tree_one = tree.DecisionTreeClassifier()

my_tree_one = my_tree_one.fit(features_one, target)

# Impute the missing value with the median

test.Fare[152] = test.Fare.median()



# Extract the features from the test set: Pclass, Sex, Age, and Fare.

test_features = test[["Pclass","Fare"]].values



# Make your prediction using the test set and print them.

my_prediction = my_tree_one.predict(test_features)

print(my_prediction)

# Create a new array with the added features: features_two

features_two = data[["PassengerId", "Fare", "SibSp", "Parch"]].values



#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two

max_depth = 10

min_samples_split = 5

my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)

my_tree_two = my_tree_two.fit(features_two, target)



#Print the score of the new decison tree

print(my_tree_two.score(features_two, target))
# Create train_two with the newly defined feature

train_two = data.copy()

train_two["family_size"] = data["SibSp"] + data["Parch"] + 1



# Create a new feature set and add the new feature

features_three = train_two[["PassengerId", "Fare", "SibSp", "Parch", "family_size"]].values



# Define the tree classifier, then fit the model

my_tree_three = tree.DecisionTreeClassifier()

my_tree_three = my_tree_three.fit(features_three, target)



# Print the score of this decision tree

print(my_tree_three.score(features_three, target))

# Import the `RandomForestClassifier`

from sklearn.ensemble import RandomForestClassifier



# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables

features_forest = data[["PassengerId", "Fare", "SibSp", "Parch"]].values



# Building and fitting my_forest

forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)

my_forest = forest.fit(features_forest, target)



# Print the score of the fitted random forest

print(my_forest.score(features_forest, target))



# Compute predictions on our test set features then print the length of the prediction vector

test_features = test[["PassengerId", "Fare", "SibSp", "Parch"]].values

pred_forest = my_forest.predict(test_features)

print(len(pred_forest))

#Request and print the `.feature_importances_` attribute

print(my_tree_two.feature_importances_)

print(my_forest.feature_importances_)



#Compute and print the mean accuracy score for both models

print(my_tree_two.score(features_two, target))

print(my_forest.score(features_forest, target))