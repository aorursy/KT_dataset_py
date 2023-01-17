# Load libraries



%matplotlib inline



import os                            # Operating system

import numpy as np                   # Linear algebra

import pandas as pd                  # Data processing, CSV file I/O (e.g. pd.read_csv)

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt      # Data visualisations

import seaborn as sns                # Data visualisations



sns.set(style='white')





# Get available file names from source folder

for dirname, _, filenames in os.walk("./Source Data/"):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Read training and test data

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')



# Check the number of rows and columns in the data

train_data.shape, test_data.shape
# Show data types of each column

train_data.dtypes
# Show a sample of random rows

train_data.sample(5)
# Show a summary of the numeric columns in the training data

train_data.describe()
# Show a summary of the numeric columns in the test data

test_data.describe()
# Define a function to check for missing values

def get_missing_summary(dataframe):

    '''Return the count and percentage of missing values in a dataframe'''

    for col in dataframe.columns:

        num_of_missing = dataframe[col].isna().sum()

        perc_of_missing = (dataframe[col].isna().sum() / len(dataframe[col])) * 100

        print('%s  --  %d missing  --  %.1f%%' % (col, num_of_missing, perc_of_missing))
# Check training data for any null / missing values

get_missing_summary(train_data)
# Check test data for any null / missing values

get_missing_summary(test_data)
# Select the name of the label column

label_col = 'Survived'
# Plot Pclass values

plt.figure(figsize=(14,4))

plt.title('Number of survivals by Pclass')

sns.countplot(x='Pclass', hue=label_col, data=train_data)
#Inspect Name values

train_data[['Name', 'Pclass', 'Fare', 'Survived']].sample(20)
# Plot Sex values

plt.figure(figsize=(14,4))

plt.title('Number of survivals by Sex')

sns.countplot(x='Sex', hue=label_col, data=train_data)
# Plot Age values

plt.figure(figsize=(14,4))

plt.title('Number of survivals by Age')

sns.distplot( train_data.loc[(train_data.Survived == 0)]["Age"], color="red", bins=20, label="Died")

sns.distplot( train_data.loc[(train_data.Survived == 1)]["Age"], color="lightgreen", bins=20, label="Survived")

plt.legend()
# Plot SibSp values

plt.figure(figsize=(14,4))

plt.title('Number of survivals by SibSp')

sns.countplot(x='SibSp', hue=label_col, data=train_data)
# Plot Parch values

plt.figure(figsize=(14,4))

plt.title('Number of survivals by Parch')

sns.countplot(x='Parch', hue=label_col, data=train_data)
#Inspect Ticket values

train_data[['Ticket', 'Name', 'Pclass', 'Fare', 'Survived']].sort_values(by='Ticket', ascending=False).head(20)
# Plot Fare values

plt.figure(figsize=(14,4))

plt.title('Number of survivals by Fare')

sns.distplot( train_data.loc[(train_data.Survived == 0)]["Fare"], color="red", bins=20, label="Died")

sns.distplot( train_data.loc[(train_data.Survived == 1)]["Fare"], color="lightgreen", bins=20, label="Survived")

plt.legend()
# Inspect Cabin values

train_data[['Cabin', 'Name', 'Pclass', 'Fare', 'Survived']].sort_values(by='Cabin', ascending=False).head(20)
# Plot Embarked values

plt.figure(figsize=(14,4))

plt.title('Number of survivals by Embarked')

sns.countplot(x='Embarked', hue=label_col, data=train_data)