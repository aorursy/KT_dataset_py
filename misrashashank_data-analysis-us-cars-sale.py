# Import libraries

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



import os
# Read dataset

base_dir = '/kaggle/input/usa-cers-dataset/'

dataset = pd.read_csv(os.path.join(base_dir, 'USA_cars_datasets.csv'))
# Explore dataset

dataset.head(10)
# Explore dataset

dataset.describe()
# Observations

# 1. Price range - 0 to 84900 - Zero price is not correct, we will have to

# handle this later

# 2. Year - 25% quartile in 2016 - Seems that more than 75% cars are new as

# they are from year 2016 and after

# 3. Mileage - 0 to 1017936 - Some cars haven't been driven - New.

# 4. Unnamed: 0 column - Just for index, can be removed.
# Column names

dataset.columns
# Column datatypes

dataset.dtypes
# Number of records with price as zero

dataset.loc[dataset['price'] == 0].count()
# Adjusting price where it is zero

# Using median to replace zero price

price_median = dataset.price.median()

dataset['price'].replace(0, price_median, inplace=True)

dataset.describe()
# Categorize variables

continuous_vars = {

    'price': dataset.price,

    'mileage': dataset.mileage,

    'year': dataset.year

}



categorical_vars = {

    'brand': dataset.brand, 

    'model': dataset.model,

    'title_status': dataset.title_status, 

    'color': dataset.color,

    'state': dataset.state, 

    'country': dataset.country, 

    'condition': dataset.condition

}

# Misc variables: 'vin', 'lot'
# Continuous variables

for name, data in continuous_vars.items():

  plt.hist(data, bins=100, color='purple')

  plt.xlabel(name)

  plt.show()



  plt.boxplot(data)

  plt.xlabel(name)

  plt.show()

# Categorical variables

# Plan to use Bar chart and Frequency table

# TBD
# Brand and Model

brand_model = dataset.groupby('brand')['model'].count()

brand_model = brand_model.reset_index().sort_values('model', ascending=False)

brand_model = brand_model.rename(columns = {'model':'count'})

fig = px.bar(brand_model, x='brand', y='count', color='count')

fig.show()
# Scatter Plot

# Variables to use - price, mileage

sns.pairplot(dataset[['price', 'mileage']],

            kind='scatter',

            diag_kind='auto')

plt.show()
# Observations:

# Price and Mileage are inversely related

# Usually, matching with the real-world scenario.