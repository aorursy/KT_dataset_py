# Import the Pandas library
import pandas as pd
# Load the train and test datasets to create two DataFrames
train_url = "../input/titanicmy-first-prediction/train.csv"
train = pd.read_csv(train_url)

test_url = "../input/titanicmy-first-prediction/test.csv"
test= pd.read_csv(test_url)
#Print the `head` of the train and test dataframes
print(train.head())
print(test.head())
train.describe()
train.shape
test.describe()
test.shape

# Passengers that survived vs passengers that passed away
print("Survived passengers vs passengers passed away:\n" , train["Survived"].value_counts())

# As proportions
print("Survived passengers vs passengers passed away:\n" , train["Survived"].value_counts(normalize=True).round(2))

# Males that survived vs males that passed away
print("Males survived: \n", train["Survived"][train["Sex"] == 'male'].value_counts())

# Females that survived vs Females that passed away
print("Females survived: \n",train["Survived"][train["Sex"] == 'female'].value_counts())

# Normalized male survival
print("Males survived: \n", train["Survived"][train["Sex"] == 'male'].value_counts(normalize=True))

# Normalized female survival
print("Females survived: \n",train["Survived"][train["Sex"] == 'female'].value_counts(normalize=True))

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
test_one=test

# Initialize a Survived column to 0
test_one['Survived'] = 0

# Set Survived to 1 if Sex equals "female" and print the `Survived` column from `test_one`
test_one['Survived'][test_one['Sex'] == 'female'] = 1
print(test_one['Survived'])




