#import all libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
#Load data

df = pd.read_csv("/kaggle/input/sales-data/sales.csv")

df
#shape of data

df.shape
#first 5 rows

df.head(5)
# last 5 records

df.tail(5)
# describe data

df.describe()
# data visualization

plt.figure(figsize=(16,9))

sns.set_style('whitegrid')

sns.countplot(x="Purchased", hue="Gender", data=df)

#plt.show()
# NAN values

sns.heatmap(df.isnull(), yticklabels=False, cmap='viridis', cbar=False)
# fill NAN values

df['Age'].fillna(df['Age'].mean(), inplace=True)

df['EstimatedSalary'].fillna(df['EstimatedSalary'].mean(), inplace=True)
#converting string values to categorical variables

gender = pd.get_dummies(df['Gender'], drop_first=True)
#add catrgorical data to main data

df = pd.concat([df,gender], axis=1)

df
# drop unncessary columns

df.drop(['User ID','Gender'], axis=1, inplace=True)
# identify dependent variable

y = df['Purchased']

y
# identify independent variables

x = df.drop(['Purchased'], axis=1)

x