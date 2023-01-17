#Data analysis

import pandas as pd

from pandas import Series,DataFrame



import numpy as np



#Graphics

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



#Machine learning

from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_recall_fscore_support

from sklearn import svm
#Read files into the program

test = pd.read_csv("../input/test.csv", index_col='PassengerId')

train = pd.read_csv("../input/train.csv", index_col='PassengerId')
print ("Basic statistical description:")

train.describe()
# absolute numbers

train["Survived"].value_counts()



# percentages

train["Survived"].value_counts(normalize = True)
# Passengers that survived vs passengers that passed away

print(train.Survived.value_counts())



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
# Create the column Child and assign to 'NaN'

train["Child"] = float('NaN')



# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.

train["Child"][train["Age"] < 18] = 1

train["Child"][train["Age"] >= 18] = 0

print(train["Child"])



# Print normalized Survival Rates for passengers under 18

print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))



# Print normalized Survival Rates for passengers 18 or older

print(train["Survived"][train["Child"] == 0].value_counts(normalize = True))
# Create a copy of test: test_one

test_one = test



# Initialize a Survived column to 0

test_one["Survived"] = 0



# Set Survived to 1 if Sex equals "female"

test_one["Survived"][test_one["Sex"] == "female"] = 1

print(test_one.Survived)