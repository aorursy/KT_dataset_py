# Import Libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Will import regressors we need from scikit at some point



import os
# Read in entire data set including all positions

full_data = pd.read_csv("../input/nfl-combine-data/combine_data_since_2000_PROCESSED_2018-04-26.csv")

full_data.head()
# Find all wide receivers

wr_data = full_data.loc[full_data['Pos'] == 'WR']

wr_data.head()
# Reset the index to start from 0

wr_data = wr_data.reset_index(drop=True)

wr_data.head()
# Setting undrafted players round to 10.0, and pick to 300.0

wr_data['Round'].fillna(10.0, inplace = True)

wr_data['Pick'].fillna(300.0, inplace = True)

wr_data.head()
# Remove columns we do not care about

# We know every player is a WR, bench press data is inconclusive, and the other columns are irrelevant

wr_data = wr_data.drop(['Pos', 'Year', 'Pfr_ID', 'AV', 'Team', 'BenchReps'], 1)

wr_data.head()
# Will need to edit the database manually at this point to fill in missing combine, and drafted round numbers

wr_data.to_csv("wide_receiver_data.csv")
# After manually filling in missing draft round and pick numbers this is our new dataset to work with

wr_data = pd.read_csv("../input/20002018-wide-receiver-combine-data/wide_receiver_data.csv")

wr_data = wr_data.drop(['Unnamed: 0'], 1)

wr_data.head()
# Now that we added in missing round numbers manually we can begin to look into the data

wr_data.describe()
# We'll start by doing some scatter plots to find any interesting information between the round drafted

# and athletic testing
# Round and Forty

sns.relplot(x="Forty", y="Round", data=wr_data)
# Round and Vertical

sns.relplot(x="Vertical", y="Round", data=wr_data)
# Round and Broad Jump

sns.relplot(x="BroadJump", y="Round", data=wr_data)
# Round and 3 Cone

sns.relplot(x="Cone", y="Round", data=wr_data)
# Round and Shuttle

sns.relplot(x="Shuttle", y="Round", data=wr_data)