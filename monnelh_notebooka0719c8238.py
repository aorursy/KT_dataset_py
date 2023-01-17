# Imports



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# pandas

import pandas as pd

from pandas import Series, DataFrame



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
# Data Import

# get the data from csv

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

    

# preview the data

train.head()
print('Train Data Infor:')

train.info()

print('---------------------------------')

print('Test Data Infor:')

test.info()
# drop unnecessary columns for the char type

train = train.drop(['PassengerId','Name','Ticket'], axis=1)
train.head()
train.info()
# fill missing value for age and embarked with mode

train['Age'] = train['Age'].fillna(train['Age'].mean())

train['Embarked'] = train['Embarked'].fillna('S')

train.head()
train.info()
train.describe()