import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
# Create dataframe

df = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
# Check columns

df.columns
df.info()
df.head()
df = df.dropna()
# Generate descriptive statistics for relevant numeric fields.

df[['price'

    ,'minimum_nights'

    ,'number_of_reviews'

    ,'reviews_per_month'

    ,'calculated_host_listings_count'

    ,'availability_365']].describe()
# Create a histogram of price

df_price = df.loc[(df['price'] >= 1) & (df['price'] <= 1000)]



sns.distplot(df_price['price']);
# Create a histogram of reviews per month

sns.distplot(df['reviews_per_month']);
# Check different room types available

df.room_type.unique()
# Create a box plot of room_type vs price

var = 'room_type'

box_data = pd.concat([df['price'], df[var]], axis=1)

f, ax = plt.subplots(figsize=(6, 6))

fig = sns.boxplot(x=var, y='price', data=box_data)

fig.axis(ymin=0, ymax=500);
# Create a box plot of neighborhood (group) vs price

var = 'neighbourhood_group'

box_data = pd.concat([df['price'], df[var]], axis=1)

f, ax = plt.subplots(figsize=(6, 6))

fig = sns.boxplot(x=var, y='price', data=box_data)

fig.axis(ymin=0, ymax=500);
# Create a box plot of neighborhood (group) vs reviews per month

var = 'neighbourhood_group'

box_data = pd.concat([df['reviews_per_month'], df[var]], axis=1)

f, ax = plt.subplots(figsize=(6, 6))

fig = sns.boxplot(x=var, y='reviews_per_month', data=box_data)

fig.axis(ymin=0, ymax=10);
# What are the most frequently listed neighborhoods?

df_neighbourhood = df['neighbourhood'].value_counts().head(20)

df_neighbourhood
# Create a box plot of top 20 neighborhoods vs price, ordered by listing frequency

var = 'neighbourhood'

box_data = pd.concat([df['price'], df[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 6))

fig = sns.boxplot(x=var

                  , y='price'

                  , data=box_data[box_data[var].isin(df_neighbourhood.index)]

                  , order = df_neighbourhood.index)

fig.axis(ymin=0, ymax=500)

fig.set_xticklabels(fig.get_xticklabels(), rotation=45, horizontalalignment='right');
# Create a box plot of neighborhood vs reviews per month, ordered by listing frequency

var = 'neighbourhood'

box_data = pd.concat([df['reviews_per_month'], df[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 6))

fig = sns.boxplot(x=var

                  , y='reviews_per_month'

                  , data=box_data[box_data[var].isin(df_neighbourhood.index)]

                  , order = df_neighbourhood.index)

fig.axis(ymin=0, ymax=10)

fig.set_xticklabels(fig.get_xticklabels(), rotation=45, horizontalalignment='right');