# modules we'll use
##Pandas is a Python package providing fast, flexible, and expressive data structures designed to make 
#working with structured (tabular, multidimensional, potentially heterogeneous) 
#and time series data both easy and intuitive (source: https://pypi.org/project/pandas/)

#NumPy is a Python library that provides a multidimensional array object, various derived objects (such as 
#masked arrays and matrices), and an assortment of routines for fast operations on arrays, including  
#mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic 
#linear algebra, basic statistical operations, random simulation and much more.
#(source: https://docs.scipy.org/doc/numpy/user/whatisnumpy.html or http://www.numpy.org/)

import pandas as pd 
import numpy as np

# read in all our data
nfl_data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")
sf_permits = pd.read_csv("../input/building-permit-applications-data/Building_Permits.csv")

# set seed for reproducibility
np.random.seed(0) 
#"A random seed specifies the start point when a computer generates a random number sequence.
#With the seed reset (every time), the same set of numbers will appear every time.
#For example, let’s say you wanted to generate a random number in Excel (Note: Excel sets a 
#limit of 9999 for the seed). If you enter a number into the Random Seed box during the process, 
#you’ll be able to use the same set of random numbers again. If you typed “77” into the box, and 
#typed “77” the next time you run the random number generator, Excel will display that same set 
#of random numbers. If you type “99”, you’ll get an entirely different set of numbers. But if you 
#revert back to a seed of 77, then you’ll get the same set of random numbers you started with. 
#(source: https://stackoverflow.com/questions/21494489/what-does-numpy-random-seed0-do)"
# look at a few rows of the nfl_data file. I can see a handful of missing data already!
nfl_data.sample(5)
# your turn! Look at a couple of rows from the sf_permits dataset. Do you notice any missing data?
sf_permits.sample(5)
# your code goes here :)
# get the number of missing data points per column
missing_values_count = nfl_data.isnull().sum()

#isnull in python: Detect missing values (NaN in numeric arrays, None/NaN in object arrays)

# look at the # of missing points in the first ten columns
missing_values_count[20:70]
# how many total missing values do we have?
total_cells = np.product(nfl_data.shape)
total_missing = missing_values_count.sum()
#total_cells
#total_missing
#numpy.prod - Return the product of array elements over a given axis.(source: https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.prod.html)
# The shape attribute for numpy arrays returns the dimensions of the array. If Y has n rows and m columns, then Y.shape is (n,m).

# percent of data that is missing
(total_missing/total_cells) * 100
# your turn! Find out what percent of the sf_permits dataset is missing

# get the number of missing data points per column
missing_values_count_p = sf_permits.isnull().sum()

# how many total missing values do we have?
total_cells_p = np.product(sf_permits.shape)
total_missing_p = missing_values_count_p.sum()
(total_missing_p/total_cells_p)*100
# look at the # of missing points in the first ten columns
missing_values_count_p[0:10]
# remove all the rows that contain a missing value
nfl_data.dropna()
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])

## The shape attribute for numpy arrays returns the dimensions of the array. If Y has n rows and m columns, then Y.shape is (n,m).
# %d acts as a placeholder for a number 
# The format operator, % allows us to construct strings, replacing parts of the strings
#with the data stored in variables. #For example, the format sequence %d means that the second operand should be
#formatted as an integer (“d” stands for “decimal”) (Source: Python for Informatics, Charles Severance: http://www.pythonlearn.com/html-270/)
#\n - The '\n' sequence is used to indicate a new line in a string
# Your turn! Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?
sf_permits.dropna()
# Now try removing all the columns with empty values. Now how much of your data is left?
columns_with_na_dropped = sf_permits.dropna(axis=1)
columns_with_na_dropped.head()
print("Columns in original dataset: %d \n" % sf_permits.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head() # all rows from the specific selection of columns
subset_nfl_data

#Series.loc - Access a group of rows and columns by label(s) or a boolean array.
# (source: https://stackoverflow.com/questions/44890713/selection-with-loc-in-python) 
#"pd.DataFrame.loc can take one or two indexers. For the rest of the post, 
#I'll represent the first indexer as i and the second indexer as j.
#If only one indexer is provided, it applies to the index of the dataframe and the 
#missing indexer is assumed to represent all columns. So the following two examples are equivalent.
#df.loc[i]
#df.loc[i, :]
#Where : is used to represent all columns.
#If both indexers are present, i references index values and j references column values.""
# replace all NA's with 0
subset_nfl_data.fillna(0)
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the reamining na's with 0
subset_nfl_data.fillna(method = 'bfill', axis=0).fillna(0)

#DataFrame.fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None, **kwargs)
#method : {‘backfill’, ‘bfill’, ‘pad’, ‘ffill’, None}, default None
#Method to use for filling holes in reindexed Series pad / ffill: propagate last valid observation forward to next valid backfill / bfill: use NEXT valid observation to fill gap
#axis : {0 or ‘index’, 1 or ‘columns’}
#source: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html
# Your turn! Try replacing all the NaN's in the sf_permits data with the one that
# comes directly after it and then replacing any remaining NaN's with 0
sf_permits.fillna(method = 'bfill', axis=0).fillna(0)