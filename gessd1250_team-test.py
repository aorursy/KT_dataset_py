import pandas as pd

import numpy as np





#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")



# Hey, Joshua.



# I will start growing through a list of commands in Pandas that we can look through... I will make certain to 

# comment them and please, ask any questions on how it works.
# It seems, at least in my perspective, the Pytohn package Seaborn is popular

# for Data Visualization... Seaborn runs on top of MatPlotLib, which is

# the main Data Visualization package for Python.

# There seems to be a recent push to a package known as Plotly...

# We will import both Seaborn and Plotly for visualization.

import seaborn as sns

import plotly.graph_objects as go





# Let's look at our data...

df = train

print('There are {} columns in the DataFrame...'.format(len(df.columns)))



# Let's see the null values...

# This command will return a Bool value for each cell depending if the value is null (na)..

null_percent = df.isna()



# This command will sum the total number of False values...

null_percent = null_percent.sum()



# This command will return the percentage of values null in the column...

null_percent = (null_percent / len(df)) * 100



# We now print the top 15 columns that have the highest null value percentage...

# Note: the ".head()" command will print the first n rows...

print(null_percent.sort_values(ascending = False).head(n = 15))
# So in terms of Feature Engineering... our largest hurdle is going to be the

# columns listed above... 

# Depending on where the housing data came from... the above might make sens.

# For example, houses in Iowa may not have many pools... Wisconsin houses are typically

# not near alleys... so, the above makes sense.

# I think it will be nice to use a HeatMap to determine if there exist any correaltions

# between the SalePrice of the home and any of our fetures (columns)... 

# The HeatMap uses the Pearson coefficient.. A simple measurement that calculates the average

# distance between a line that best fits the feature and target value...

print('It looks like OverallQual, GrLivArea, GarageCars, GarageArea, and TotalBsmtSF\nhave the greatest positive correlation to SalePrice.')

print(sns.heatmap(df[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', 'SalePrice']].corr()))

print(df.corr()['SalePrice'].sort_values(ascending = False).head(n=6))