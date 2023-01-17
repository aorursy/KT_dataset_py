import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/surviving-mars-maps/MapData-Evans-GP-Flatten.csv', skipinitialspace=True)

totalRows = len(df)

print(str(totalRows) + " rows loaded.")
# let's look at the names of all the columns

column_names = df.columns

print(column_names)
for icol in range(len(column_names)):

    print(column_names[icol])

    print( df[column_names[icol]].unique() )

    print('')
# get the names of all boolean type columns

bool_cols = df.select_dtypes(include=['bool']).columns
# a new column for the True counts of the bool columns in each row

df['bool True count'] = df.loc[:,bool_cols].sum(axis=1)

df.head(5)
# get the indexes of "Relatively Flat" rows

idx = np.where(df['Topography']=="Relatively Flat")

# the number of "Relatively Flat " entries

nb_rflat = len(idx[0])

# get the average boolean True count for all "Relatively Flat" entries

df.loc[idx]['bool True count'].values.sum()/nb_rflat
idx = np.where(df['Topography']=="Steep")

nb_steep = len(idx[0])

df.loc[idx]['bool True count'].values.sum()/nb_steep
idx = np.where(df['Topography']=="Rough")

nb_rough = len(idx[0])

df.loc[idx]['bool True count'].values.sum()/nb_rough
idx = np.where(df['Topography']=="Mountainous")

nb_mount = len(idx[0])

df.loc[idx]['bool True count'].values.sum()/nb_mount