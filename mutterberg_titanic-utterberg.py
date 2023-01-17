import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Load the train and test datasets to create two DataFrames
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#Print the `head` of the train and test dataframes
print(train.head())
print(test.head())
train.describe()
print(train.shape)
test.describe()
print(test.shape)
# absolute numbers
# train["Survived"].value_counts()
# percentages
# train["Survived"].value_counts(normalize = True)

# Passengers that survived vs passengers that passed away
print(train["Survived"].value_counts())

# As proportions
print(train["Survived"].value_counts(normalize = True))

# train["Survived"][train["Sex"] == 'male'].value_counts()
# train["Survived"][train["Sex"] == 'female'].value_counts()

# Males that survived vs males that passed away
print(train["Survived"][train["Sex"] == 'male'].value_counts())

# Females that survived vs Females that passed away
print(train["Survived"][train["Sex"] == 'female'].value_counts())

# Normalized male survival
print(train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True))

# Normalized female survival
print(train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True))