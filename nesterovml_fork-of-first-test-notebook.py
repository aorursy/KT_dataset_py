# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Import the `RandomForestClassifier`

from sklearn.ensemble import RandomForestClassifier

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Load the train and test datasets to create two DataFrames

train_url = "../input/train.csv"

train = pd.read_csv(train_url)



test_url = "../input/test.csv"

test = pd.read_csv(test_url)



# Create the target numpy array: target

target = train["Survived"].values



#print(train["Sex"][train["Sex"] == "male"])
# Convert the male and female groups to integer form

train["Sex"][train["Sex"] == "male"] = 0

train["Sex"][train["Sex"] == "female"] = 1



test["Sex"][test["Sex"] == "male"] = 0

test["Sex"][test["Sex"] == "female"] = 1



# Impute the Embarked variable

train["Embarked"] = train["Embarked"].fillna("S") 

train["Age"] = train["Age"].fillna(29)



test["Embarked"] = test["Embarked"].fillna("S") 

test["Age"] = test["Age"].fillna(29) 

test.Fare[152] = test["Fare"].median()



# Convert the Embarked classes to integer form

train["Embarked"][train["Embarked"] == "S"] = 0

train["Embarked"][train["Embarked"] == "C"] = 1

train["Embarked"][train["Embarked"] == "Q"] = 2



test["Embarked"][test["Embarked"] == "S"] = 0

test["Embarked"][test["Embarked"] == "C"] = 1

test["Embarked"][test["Embarked"] == "Q"] = 2
train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
sum(test["Fare"].value_counts())


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







# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions

PassengerId =np.array(test["PassengerId"]).astype(int)

my_solution = pd.DataFrame(pred_forest, PassengerId, columns = ["Survived"])

print(my_solution)



# Check that your data frame has 418 entries

print(my_solution.shape)



# Write your solution to a csv file with the name my_solution.csv

my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])
submission = my_solution
submission.to_csv('submission.csv', index_label = ["PassengerId"])
print(check_output(["ls", "../working"]).decode("utf8"))