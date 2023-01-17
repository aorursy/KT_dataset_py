
import numpy as np 
import pandas as pd


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

raw_data = pd.read_csv('/kaggle/input/titanic/train.csv') 
raw_data.head()
raw_data.describe(include='all')
# Looking at the Dataset provided, we need to preprocess the data before we can use it to train any Machine Learning algorithm, 
# First we drop the id, Name, Ticket and Cabin columns

# The id column is not expected to hace any predictive power so it will not be part of our inputs for this dataset
# The Name column also follows suit
# The Ticket and Cabiun data contain is too unique for us to create as many dummies as that number and will not also be part
# of the inputs in this dataset

df = raw_data.copy() # Creating a checkpoint here
df.head()
df = df.drop(['PassengerId', 'Name', 'Ticket','Cabin'], axis=1)
df
# Next, we map the Sex column to 0s and 1s for Male and Female respectively as they are categorical variables

df_mapped = df.copy()
df_mapped['Sex'] = df_mapped['Sex'].map({'male':0, 'female':1})
df_mapped
# We drop other data that do not have a value

df_mapped = df_mapped.dropna(axis=0)
# The Embarked colums represents where the passenger embarked from. These are Categorical nominal variables so we create
# dummy variables for them. We will drop the first column which will represent the default embarkation point. We do this to
# avoid multicollinearity during regression analysis. 

embarked_dummies = pd.get_dummies(df_mapped['Embarked'], drop_first=True, prefix = 'Embarked')
pd.set_option('max_rows', None)
embarked_dummies # The default Embarkation point here is Cherbourg

# Next we concatenete the dummies with the dataset, but before we do that, we drop the Embarked column

df_mapped = df_mapped.drop(['Embarked'], axis=1)
df_mapped
df_with_dummies = pd.concat((df_mapped,embarked_dummies), axis=1)
df_with_dummies
df_dummies = df_with_dummies.copy()
df_dummies
# The Ticket class does not have a numerical meaning as it is also a categorical(ordinal) variable. Therefore we will map
# the values to 0 and 1. With 0 representing the lower class and 1 representing the upper class.

# First, we chek the number of unique entries in the Ticket class

pd.unique(df_dummies['Pclass'])
df_mapped_class = df_dummies.copy()
df_mapped_class['Pclass'] = df_mapped_class['Pclass'].map({1:0, 2:1, 3:1})
df_mapped_class
# For organization purpose when the modelling will be done, we put the input columns to the left and the target to the right

df_mapped_class.columns.values
cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S', 'Survived']
data_cleaned = df_mapped_class[cols]
data_cleaned
data_preprocessed = data_cleaned.copy()
data_preprocessed.to_csv('titanic_preprocessed.csv', index=False)
