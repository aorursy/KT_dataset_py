import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Load the three data files into dataframes

klaverns = pd.read_csv('../input/klavern.csv')

sources = pd.read_csv('../input/sources.csv')

states = pd.read_csv('../input/states.csv')
klaverns.head()
sources.head()
# Make the column names match the of DF column names for merging

states.rename(columns={"id": "state_id"}, inplace=True)

states.head()
# Merge the three dataframes

df = states.merge(klaverns, on="state_id").merge(sources, on="id")



# Remove id

df.drop("id", axis=1, inplace=True)



df.head(10)