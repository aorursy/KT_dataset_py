

# Exploratory Data Analysis (EDA) on the Titanic dataset

# How many survivors?

# What are the splits on passenger classes?

# What are the ages and gender of survivors and non survivors?

# Is there a correlation between age and survival, gender and survival or passenger class and survival?

# What patterns and trends can be seen?
# Import packages for calcs & visualisations

import numpy as np

import pandas as pd

import re



# For Box-Cox Transformation

from scipy import stats



# for min_max scaling

from mlxtend.preprocessing import minmax_scaling



# Data visualization

import seaborn as sns

import missingno

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
# Load titanic dataset and display the 1st 5 rows

df = pd.read_csv('../input/titanic/Titanic.csv')

df.head()
# Display summarised data re dataframe

df.info()
# View Features

df.columns
# printing summary statistics

df.describe()
# Visualise missing data

missingno.matrix(df, figsize = (13,3))
# The percentage of missing values across the dataset

df.isnull().sum()/ len(df) *100
# Number of missing data points per column

missing_values_count = df.isnull().sum()



# The no of missing points in the first ten columns

missing_values_count[0:12]
# Create two new dataframes for purposes of analysis

# for discretised continuous variables- numerical

df_bin = df



# for continuous variables - categorical

df_con = df
# How many passengers were Male and Female

fig = plt.figure(figsize=(12,1))

sns.countplot(y='Sex', data=df_bin);

print(df_bin.Sex.value_counts())
# Passenger class split on the ship

fig = plt.figure(figsize=(13,1))

sns.countplot(y=df_bin['Pclass'], data=df_bin);
# Age of passengers

df.Age.plot.hist()
# Line graph for a detailed split across Age of passengers

test = df['PassengerId'].groupby(df['Age']).size()

plot_month = test.plot(title = 'Passenger split on Age', xticks = (range(10,100,10)))
# Gender of passengers plotted against Survived

sns.barplot(x='Sex',y='Survived',data=df)

df.groupby('Sex',as_index=False).Survived.mean()
# Pclass feature against Survived

sns.barplot(x='Pclass',y='Survived',data=df)

df[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Visualise age of survivors

# for discretised continuous variables ()

df_bin = df



fig = plt.figure(figsize=(5, 5))

for i in range(0, 7):

    sns.distplot(df_bin.loc[df_bin['Survived'] == i]['Age'], kde_kws={'label': f'Survived = {i}'});
# How many females do we have information for vs males?

df.groupby(['Sex']).count()
# How many Survived do we have information for vs males?

df.groupby(['Survived']).count()
# Drop columns - Cabin, Ticket, PassengerId

df.drop(['Cabin', 'Ticket', 'PassengerId'], axis=1)
# Most people embarked on their journey from Southhampton port(s)



# Fill the missing Embarked values as S

df.Embarked.fillna('S',inplace=True)
# Replace the NaN values in the Age column with the median



# Fill the missing values in the Age column

df.Age.fillna(28, inplace=True)

df.Age.fillna(28, inplace=True)
# Relook at the percentage of missing values

df.isnull().sum()/ len(df) *100