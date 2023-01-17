# import libraries we'll need

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
# read in the required dataset

df = pd.read_csv('../input/epi_r.csv')

df.head()
# quickly clean the dataset

# remove outliers

df = df[df['calories'] < 10000]



# remove rows with null values

df.dropna(inplace=True)
# are the ratings all numeric?

print('Is this variable numeric?')

df['rating'].dtype == 'float'
# are the ratings all integers?

print('Is this variable only integers?')

df['rating'].dtype == 'int'
# plot calories by whether or not it's a dessert

# fit_reg = False to turn off default regression line

sns.regplot(df['calories'], df['dessert'], fit_reg=False)
# plot calories by whether or not it's a dessert with regression line

sns.regplot(df['calories'], df['dessert'])