import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

from sklearn import metrics

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Let's load our csv data into DataFrame

df = pd.read_csv("/kaggle/input/insurance-premium-prediction/insurance.csv")
# Get an understanding of the columns and rows

df.info()
# Take a peek into data

df.head()
# Let's check for nulls first

df.isnull().any().any()
df.age.unique()
df.sex.unique()
df.sex.replace({'male':1, 'female':0}, inplace=True)
df.bmi.describe()
df.children.unique()
df.smoker.unique()
df.smoker.replace({'yes':1, 'no':0}, inplace=True)
df.region.unique()
# Using Pandas get_dummies(), we can those new dummy columns.

# After that we dont need the original region column, dropping it.

# Concatenating the new dummy columns to the exisiting dataframe.

dummies = pd.get_dummies(data=df['region'], drop_first=True).rename(columns=lambda x: 'region_' + str(x))

df.drop(['region'], inplace=True, axis=1)

df = pd.concat([df, dummies], axis=1)
df.expenses.describe()
sns.boxplot(y=df.expenses)
df.expenses = df.expenses[df.expenses<50000]
sns.boxplot(y=df.expenses)
df.dropna(inplace=True)
df.info()
df.head()
x = df[df.columns[df.columns != 'expenses']]

y = df.expenses
# Statsmodels.OLS requires us to add a constant.

x = sm.add_constant(x)

model = sm.OLS(y,x)

results = model.fit()

print(results.summary())
x.drop('sex',axis=1, inplace=True)

model = sm.OLS(y,x)

results = model.fit()

print(results.summary())
x.drop('region_northwest',axis=1, inplace=True)

model = sm.OLS(y,x)

results = model.fit()

print(results.summary())
