import numpy as np

import pandas as pd

import seaborn as sns

# Load the standardized data

# index_col is an argument we can set to one of the columns

# this will cause one of the Series to become the index

data = pd.read_csv(r'../input/heatmaps-and-dendograms-data/Country clusters standardized.csv', index_col='Country')

data.head()
# Create a new data frame for the inputs, so we can clean it

x_scaled = data.copy()

# Working with 2 features

x_scaled = x_scaled.drop(['Language'],axis=1)
# Check what's inside

x_scaled
# Using the Seaborn method 'clustermap' we can get a heatmap and dendrograms for both the observations and the features

sns.clustermap(x_scaled, cmap='rainbow');
sns.clustermap(data, cmap='Paired');