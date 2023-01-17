# Import the Pandas library

import pandas as pd

# Load the train and test datasets to create two DataFrames

train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"

train = pd.read_csv(train_url)



test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"

test = pd.read_csv(test_url)

#Print the `head` of the train and test dataframes

print(train.head())

print(test.head())
# Passengers that survived vs passengers that passed away

print(train["Survived"].value_counts())

print(train["Survived"][train["Survived"] == 0].value_counts())

# As proportions

print(train['Survived'].value_counts(normalize=True))

# Males that survived vs males that passed away

print(train["Survived"][train["Sex"] == 'male'].value_counts())



# Females that survived vs Females that passed away

print(train["Survived"][train["Sex"] == 'female'].value_counts())



# Normalized male survival

print(train["Survived"][train["Sex"] == 'male'].value_counts(normalize=True))



# Normalized female survival

print(train["Survived"][train["Sex"] == 'female'].value_counts(normalize=True))
# Create the column Child and assign to 'NaN'

train["Child"] = float('NaN')



# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.

train["Child"][train["Age"]<18] = 1

train["Child"][train["Age"]>=18] = 0



print(train['Child'])



# Print normalized Survival Rates for passengers under 18

print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))



# Print normalized Survival Rates for passengers 18 or older

print(train['Survived'][train['Child']==0].value_counts(normalize = True))
# Import the Numpy library

import numpy as np

# Import 'tree' from scikit-learn library

from sklearn import tree
# Convert the male and female groups to integer form



train["Sex"][train["Sex"] == "male"] = 0

train["Sex"][train["Sex"] == "female"] = 1

# Impute the Embarked variable

train["Embarked"] = train["Embarked"].fillna('S')



# Convert the Embarked classes to integer form

train["Embarked"][train["Embarked"] == "S"] = 0

train["Embarked"][train["Embarked"] == "C"] = 1

train["Embarked"][train["Embarked"] == "Q"] = 2

#Print the Sex and Embarked columns

print(train['Sex'])

print(train['Embarked'])