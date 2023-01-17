# Import Libaries

import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import xgboost
# Load Data

df = pd.read_csv("../input/shoppingdata/shopping-data.csv",index_col='CustomerID')

df.head(10)
# Describe standard statistics of data of numerical data

df.describe()
# Check unique fields in categorical data

df.Genre.unique()
# Find Correlation of Spending Score with

df.corr()
df_male = df.loc[df.Genre == "Male"].drop("Genre",axis=1)

df_male.corr()
df_male.describe()
df_female = df.loc[df.Genre == "Female"].drop("Genre",axis=1)

df_female.corr()
df_female.describe()
# Label Encode and replace the Genre value.

le_gender = LabelEncoder()

df.Genre = le_gender.fit_transform(df.Genre)
df.head()
X = df.iloc[:,:-1]

X.head()
y = df.iloc[:,-1].values

y