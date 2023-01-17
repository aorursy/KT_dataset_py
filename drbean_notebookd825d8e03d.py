# Import the Pandas library

import pandas as pd

# Load the train and test datasets to create two DataFrames



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

#Print the `head` of the train and test dataframes

import numpy as np

train.describe()
test.describe()
null
# Convert the male and female groups to integer form

train["Sex"][train["Sex"] == "male"] = 0

train["Sex"][train["Sex"] == "female"] = 1

# Impute the Embarked variable

train["Embarked"] = train["Embarked"].fillna("S")



# Convert the Embarked classes to integer form

train["Embarked"][train["Embarked"] == "S"] = 0

train["Embarked"][train["Embarked"] == "C"] = 1

train["Embarked"][train["Embarked"] == "Q"] = 2