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
# Import the Pandas library

import pandas as pd



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

# Create the column Child and assign to 'NaN'

train["Child"] = float('NaN')



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

print('Titanic: First Submission: Females Survive')

print(prediction.head())
prediction.to_csv('FemalesSurvive.csv', index=False)