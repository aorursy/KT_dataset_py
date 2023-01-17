# Importing required libraries.

import pandas as pd

import numpy as np

import seaborn as sns #visualisation

import matplotlib.pyplot as plt #visualisation

%matplotlib inline 

sns.set(color_codes=True)

df = pd.read_csv('/kaggle/input/FloodsDamage.csv')

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
sns.barplot(y="Lives Lost (in Nos.)", x="Year" , data=df);
sns.boxplot(x="Year" , y="Cattle Lost (in Nos.)" , data=df);
sns.barplot(x="Year" , y="House damaged (in Nos.)" , data=df);
sns.barplot(x="Year" , y="Cropped areas affected (in lakh ha)" , data=df);
sns.violinplot(x="House damaged (in Nos.)", y="Cropped areas affected (in lakh ha)", data=df)