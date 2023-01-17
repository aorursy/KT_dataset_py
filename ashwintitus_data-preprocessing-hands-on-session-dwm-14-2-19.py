# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



# Importing required data



nfl_data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")
# look at a few rows of the nfl_data file.

nfl_data.sample(5)

# Missing values are represented by NaN

# get the number of missing data points per column

missing_values_count = nfl_data.isnull().sum()



# look at the # of missing points in the first ten columns

missing_values_count[0:10]
# how many total missing values do we have?

total_cells = np.product(nfl_data.shape)

total_missing = missing_values_count.sum()



# percent of data that is missing

(total_missing/total_cells) * 100
# remove all the rows that contain a missing value

nfl_data.dropna()
# remove all columns with at least one missing value

columns_with_na_dropped = nfl_data.dropna(axis=1)

columns_with_na_dropped.head()
# just how much data did we lose?

print("Columns in original dataset: %d \n" % nfl_data.shape[1])

print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# get a small subset of the NFL dataset

subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()

subset_nfl_data
# replace all NA's with 0

subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 

# then replace all the reamining na's with 0

subset_nfl_data.fillna(method = 'bfill', axis=0).fillna(0)
# Your turn! Try replacing all the NaN's in the nfl_data data with the one that

# comes directly after it and then replacing any remaining NaN's with 0
nfl_predictors = nfl_data.select_dtypes(exclude=['object'])

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

data_with_imputed_values = my_imputer.fit_transform(nfl_predictors)

# Default behavior is filling with mean values

data_with_imputed_values
# modules we'll use

import pandas as pd

import numpy as np



# for Box-Cox Transformation

from scipy import stats



# for min_max scaling

from mlxtend.preprocessing import minmax_scaling



# plotting modules

import seaborn as sns

import matplotlib.pyplot as plt



# read in all our data

kickstarters_2017 = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")

# generate 1000 data points randomly drawn from an exponential distribution

original_data = np.random.exponential(size = 1000)



# mix-max scale the data between 0 and 1

scaled_data = minmax_scaling(original_data, columns = [0])



# plot both together to compare

fig, ax=plt.subplots(1,2)

sns.distplot(original_data, ax=ax[0])

ax[0].set_title("Original Data")

sns.distplot(scaled_data, ax=ax[1])

ax[1].set_title("Scaled data")
df = pd.DataFrame(np.random.randint(100, 200, size=(5, 3)), columns=['A', 'B', 'C'])

print(df)

from scipy.stats import zscore

df.apply(zscore)