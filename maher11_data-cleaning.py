# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



nfl_data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")



# set seed for reproducibility

np.random.seed(0) 

nfl_data.head()
# get the number of missing data points per column

missing_values_count = nfl_data.isnull().sum()



# look at the # of missing points in the first ten columns

missing_values_count[0:10]
# how many total missing values do we have?

total_cells = np.product(nfl_data.shape)

total_missing = missing_values_count.sum()



# percent of data that is missing

percent_missing = (total_missing/total_cells) * 100

print(percent_missing)
"""1- Drop the missing data"""



# remove all the rows that contain a missing value

nfl_data.dropna()
# remove all columns with at least one missing value

columns_with_na_dropped = nfl_data.dropna(axis=1)

columns_with_na_dropped.head()
# just how much data did we lose?

print("Columns in original dataset: %d \n" % nfl_data.shape[1])

print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
"""2- Filling in missing values automatically"""

#Another option is to try and fill in the missing values. For this next bit, I'm getting a small sub-section of the NFL data so that it will print well.

# get a small subset of the NFL dataset

subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()

subset_nfl_data
# replace all NA's with 0

subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 

# then replace all the remaining na's with 0

subset_nfl_data.fillna(method='bfill', axis=0).fillna(0)