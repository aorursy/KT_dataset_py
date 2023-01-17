# libraries that we'll need

import numpy as np 

import pandas as pd



# read in all three datasets (you'll pick one to use later)

recipes = pd.read_csv("../input/epirecipes/epi_r.csv")

bikes = pd.read_csv("../input/nyc-east-river-bicycle-crossings/nyc-east-river-bicycle-counts.csv")

weather = pd.read_csv("../input/szeged-weather/weatherHistory.csv")
# quickly clean our dataset

recipes = recipes[recipes['calories'] < 10000].dropna()
# are the ratings all numeric?

from pandas.api.types import is_numeric_dtype

print("Is this variable numeric?")

#np.issubdtype(recipes['rating'].dtype, np.number)

is_numeric_dtype(recipes['rating'])
# are the ratings all integers?

print("Is this variable only integers?")

np.issubdtype(recipes['rating'].dtype, np.int64)
# plot calories by whether or not it's a dessert

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='darkgrid', palette='Set1')



plt.figure(figsize=(8,8))

sns.regplot(x='calories', y='dessert', data=recipes, fit_reg=False)

plt.show()
# plot & add a regression line

plt.figure(figsize=(8,8))

sns.regplot(x='calories', y='dessert', data=recipes, logistic=True)

plt.show()
# first lets check the head of our data set to se what we are dealing with

weather.head()