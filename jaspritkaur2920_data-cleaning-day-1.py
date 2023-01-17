import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv')
df.head()
# used to generate same sample everytime we run the code

np.random.seed(0) 



# used to generate a sample random row or column

df.sample(5) 
# get the number of missing values per column

missing = df.isnull().sum()



# look at the number of missing values in first 10 rows

missing[0:10]
# How many total missing values we have?

total_cells = np.product(df.shape)

total_missing = missing.sum()



# percent of data that is missing

(total_missing / total_cells) * 100
# look at the # of missing points in the first ten columns

missing[0:10]
df['PenalizedTeam'].isnull().value_counts()
# removes all the rows that have missing values

df.dropna()
# remove all columns with atleast one missing value

columns_with_na_dropped = df.dropna(axis=1)

columns_with_na_dropped
# just how much data did we lose?

print("Columns in original dataset: %d \n" % df.shape[1])

print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# get a small subset of NFL dataset

subset_df = df.loc[:, 'EPA' : 'Season'].head()

subset_df
subset_df.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 

# then replace all the reamining na's with 0

subset_df.fillna(method = 'bfill', axis=0).fillna(0)