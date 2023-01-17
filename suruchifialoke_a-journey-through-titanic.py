# Imports

# pandas and numpy

import pandas as pd

from pandas import Series,DataFrame

import numpy as np
# Loading data and printing first few rows

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

# Previewing the statistics of training data and test data

test.head(2)
# print the names of the columns in the data frame

print("In Training Data missing, columns with missing values:")

# Retain columns that are of interest and discard the rest (such as Id, Name, Cabin, and Ticket number)

newcols = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare','Embarked']

train = train[newcols]

# Check which columns have missing data

for column in train.columns:

    if np.any(pd.isnull(train[column])) == True:

        print(column)  
# print the names of the columns in the data frame

print("In test Data missing, columns with missing values:")

# Retain columns that are of interest and discard the rest (\Name, Cabin and Ticket number)

newcols = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare', 'Embarked']

test = test[newcols]

# Check which columns have missing data

for column in test.columns:

    if np.any(pd.isnull(test[column])) == True:

        print(column)
for df in [train, test]:

    df['FamilySize'] = df['Parch'] + df['SibSp'] + 1



def filter_family_size(x):

    if x == 1:

        return 'Solo'

    elif x < 4:

        return 'Small'

    else:

        return 'Big'



for df in [train, test]:

    df['FamilySize'] = df['FamilySize'].apply(filter_family_size)
# matplotlib and seaborn

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('white')

%matplotlib inline

plt.rcParams.update({'font.size': 22})
# Check with Pclass and Embarked

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,4))

ax1.set_title('Survival rates for Passenger Classes')

sns.barplot(x='Pclass', y='Survived', hue='Sex', data=train, ax=ax1)



ax2.set_title('Survival rates for Port Embarked')

sns.barplot(x='Embarked', y='Survived', hue='Sex', data=train, ax=ax2)

sns.despine()

sns.set(font_scale=1.4)

ax2.legend_.remove()

ax1.legend(loc='upper right')

plt.show()
# Check with Age

g = sns.FacetGrid(train, col="Survived", hue='Sex', size=5, aspect = 1.2)

g.map(sns.kdeplot, "Age", shade=True).add_legend().fig.subplots_adjust(wspace=.3)

sns.despine()

sns.set(font_scale=2)

plt.show()
# Check with Fare

g = sns.FacetGrid(train, col="Survived", hue='Sex', size=5, aspect = 1.2)

g.map(sns.kdeplot, "Fare", shade=True).add_legend().fig.subplots_adjust(wspace=.3)

sns.set(font_scale=2)

sns.despine()

plt.show()
# Family Size

sns.barplot(x='FamilySize', y='Survived' , data=train, order = ['Solo', 'Small', 'Big'])

sns.set(font_scale=1.5)

plt.show()
# Convert to numeric values

train.loc[train["Embarked"] == 'S', "Embarked"] = 0

train.loc[train["Embarked"] == 'C', "Embarked"] = 1

train.loc[train["Embarked"] == 'Q', "Embarked"] = 2



test.loc[test["Embarked"] == 'S', "Embarked"] = 0

test.loc[test["Embarked"] == 'C', "Embarked"] = 1

test.loc[test["Embarked"] == 'Q', "Embarked"] = 2
# convert female/male to numeric values (male=0, female=1)

train.loc[train["Sex"]=="male","Sex"]=0

train.loc[train["Sex"]=="female","Sex"]=1



test.loc[test["Sex"]=="male","Sex"]=0

test.loc[test["Sex"]=="female","Sex"]=1
# convert family size to numeric values



train.loc[train["FamilySize"] == 'Solo', "FamilySize"] = 0

train.loc[train["FamilySize"] == 'Small', "FamilySize"] = 1

train.loc[train["FamilySize"] == 'Big', "FamilySize"] = 2



test.loc[test["FamilySize"] == 'Solo', "FamilySize"] = 0

test.loc[test["FamilySize"] == 'Small', "FamilySize"] = 1

test.loc[test["FamilySize"] == 'Big', "FamilySize"] = 2