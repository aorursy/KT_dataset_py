# Importing required libraries.

import pandas as pd

import numpy as np

import seaborn as sns #visualisation

import matplotlib.pyplot as plt #visualisation

%matplotlib inline 

sns.set(color_codes=True)

df = pd.read_csv('/kaggle/input/CrimeAgainstWomen.csv')

# To display the top 5 rows

df.head(5)
# To display the bottom 5 rows

df.tail(5)
# Checking the data type

df.dtypes
df.columns
# Total number of rows and columns

print(df.shape)
# Used to count the number of rows 

df.count()
# Finding the null values.

print(df.isnull().sum())