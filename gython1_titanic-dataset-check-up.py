# Import libraries

import pandas as pd

import seaborn as sns

import numpy as np

import os

import re

from sklearn import preprocessing



# set directory

os.chdir('../input')



# import data

dataset_train = pd.read_csv('train.csv')
# Check dataset

dataset_train.head()
# Good way to describe numeric variables in dataset

dataset_train.describe()
# Check type of columns

dataset_train.dtypes
# Check missing values

dataset_train.isnull().sum()
# Titles from names

dataset_train["Titles"] = dataset_train["Name"].map(lambda name: name.split(',')[1].split('.')[0].strip())



# Check the occurence of titles

dataset_train["Titles"].value_counts()
# A map of more aggregated titles

Title_Dictionary = {

                    "Capt":       "Officer",

                    "Col":        "Officer",

                    "Major":      "Officer",

                    "Jonkheer":   "Royalty",

                    "Don":        "Royalty",

                    "Sir" :       "Royalty",

                    "Dr":         "Officer",

                    "Rev":        "Officer",

                    "the Countess":"Royalty",

                    "Dona":       "Royalty",

                    "Mme":        "Mrs",

                    "Mlle":       "Miss",

                    "Ms":         "Mrs",

                    "Mr" :        "Mr",

                    "Mrs" :       "Mrs",

                    "Miss" :      "Miss",

                    "Master" :    "Master",

                    "Lady" :      "Royalty"

                    }



# We map each title

dataset_train['Titles'] = dataset_train['Titles'].map(Title_Dictionary)



dataset_train["Titles"].value_counts()
# Extract only surname from name 

dataset_train["Surname"] = dataset_train["Name"].map(lambda name: name.split(',')[0].strip())



# Create pivot table surname and ticket

dataset_pivot = pd.pivot_table(dataset_train, index=["Surname", "Ticket"], values=["Name"], aggfunc='count')



dataset_pivot
# Delete unnecessary columns

dataset_train = dataset_train.drop('Name', 1)
### Age



# Create categories from age

dataset_train["Agec"] = pd.cut(dataset_train["Age"], [0,20,40,60,100], labels=['from 0 to 20','from 21 to 40','from 41 to 60','from 61 to 100'])



# Replace missing values at age variable with means of age, calculated by titles (as age is quite different among different groups by titles)



dataset_train['Age'] = dataset_train["Age"].fillna(dataset_train.groupby("Titles")["Age"].transform("mean"))
### Ticket

# Transform ticket to numeric variable, caouse I don't know what else I could do with it, I guess maybe checking it up with cabin in something interesting appears



le = preprocessing.LabelEncoder()



le.fit(dataset_train["Ticket"])



dataset_train["Ticket"] = le.transform(dataset_train["Ticket"]) 
### Cabin

# Transform cabin to numeric variable - I think best option is to factorize, however -1 has to be changed later to NaN



dataset_train['Cabin'] = dataset_train['Cabin'].factorize()[0]



dataset_train['Cabin'] = dataset_train['Cabin'].where(dataset_train['Cabin'] > -1)
### Embarked

# Transform cabin to numeric variable - I think best option is to factorize, however -1 has to be changed later to NaN



dataset_train['Embarked'] = dataset_train['Embarked'].factorize()[0]



# Keep coding frame

embarked_mapping = {"Q": 2, "C": 1, "S": 0}
### Check correlations

corr = dataset_train.corr(method='pearson')



print(corr)



# Apparently cabin has nothing to do with Survival, so why don't we just drop it, there is too much missing values anyway

dataset_train = dataset_train.drop('Cabin', 1)             
# Also repair category agec variable

dataset_train["Agec"] = pd.cut(dataset_train["Age"], [0,20,40,60,100], labels=['from 0 to 20','from 21 to 40','from 41 to 60','from 61 to 100'])             



dataset_pivot = pd.pivot_table(dataset_train, index=["Agec"], values=["Gender"], columns=['Survived'], aggfunc='count')

             

ay = sns.heatmap(dataset_pivot, annot=True, linewidths=.5, fmt='d')
# To be continued...