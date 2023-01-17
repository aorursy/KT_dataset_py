import numpy as np 

import pandas as pd 

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



import os

print(os.listdir("../input"))
sftree = pd.read_csv("../input/san_francisco_street_trees.csv", header=0)

sftree.columns = ['Address', 'Care_assistant', 'Care_taker', 'dbh', 'Latitude', 'Legal_status', 'Location', 'Longitude', 'Permit_notes', 'Plant_date', 'Plant_type', 'Plot_size', 'Site_info', 'Site_order', 'Species', 'Tree_ID', 'X_coordinate', 'Y_coordinate']

sftree.head()
sftree.shape
# Examine data

print(len(sftree), " rows x ", len(sftree.columns), " columns")

print(sftree.columns)

sftree.info()

sftree.describe()
# Make a copy of the data to work with.

sf_trees = sftree.copy()
# Clean data

sf_trees = sf_trees.loc[sf_trees['Latitude'] < 38.000000] # For some reason this goes all the way up to Seattle??

sf_trees = sf_trees.loc[sf_trees['Species'] != 'Tree(s) ::'] # Clean this bit of data
# This looks much more reasonable

sf_trees.describe()
# Count trees by species

species_count = sf_trees.groupby('Species').Tree_ID.count()

print("There are", len(species_count), "species of tree in San Francisco.")
# Top 10 tree species in SF bar chart

top_ten_species = species_count.nlargest(10)

ax = top_ten_species.plot.bar()
# Top 10 tree species in SF pie chart

ax = top_ten_species.plot.pie()
# Get dataframe where only popular trees are shown

popular_species = top_ten_species.index.tolist()

popular_species_df = sf_trees.loc[sf_trees['Species'].isin(popular_species)]

# display(popular_species_df)
# Basic scatterplot of the top ten most popular trees in SF.

x, y = popular_species_df.Latitude.tolist(), popular_species_df.Longitude.tolist()

plt.scatter(x,y)
sns.lmplot(x="Latitude", y="Longitude", col="Species", data=popular_species_df,

          col_wrap=2, lowess=True)