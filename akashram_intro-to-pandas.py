import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))
data = pd.read_csv("../input/winemag-data_first150k.csv")
data.head(10)
s = pd.Series(data['points'])
s
type(s)
dates = pd.date_range('20130101', periods=6)
dates
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
df
data.dtypes
data.head(5)
data.tail(5)
#### Display the index, columns, and the underlying NumPy data:

data.index
# columns

data.columns
# values

data.values
# check stats of the numerical columns

data.describe()
# Transposing your data:

data.T
# Sorting by an axis:

data.sort_index(axis=1, ascending=False)
# Sorting by values:

data.sort_values(by='country')
# Selecting a single column, which yields a Series, equivalent to df.A:

data['province'].head(5)  # head(5) to print only the top 5 rows
# Selecting via [], which slices the rows.

data[0:3] # Python does not consider the last index so in this case 3 and therefore prints rows 0,1,2
# Selection by Label

data.loc[0]
# Selecting on a multi-axis by label:

data.loc[:,['province','winery']].head(5)
# Select via the position of the passed integers:

data.iloc[3]
# By integer slices, acting similar to numpy/python:

data.iloc[3:5,0:2]
# By lists of integer position locations, similar to the numpy/python style:

data.iloc[[1,2,4],[0,2]]
# For slicing rows explicitly:

data.iloc[1:3,:]
#For slicing columns explicitly:

data.iloc[:,1:3].head(4)
# For getting a value explicitly:

data.iloc[1,1]
# For getting fast access to a scalar (equivalent to the prior method):

data.iat[1,1]
# Using a single column’s values to select data.

data[data.price > 100].head(5)
# Selecting values from a DataFrame where a boolean condition is met.

data[data > 0]
# Using the isin() method for filtering:

data[data['country'].isin(['US', 'Spain'])].head(5)
#  To drop any rows that have missing data.

data.dropna(how='any')
# Filling missing data.

data.fillna(value=5)
# To get the boolean mask where values are nan.

pd.isna(data)
#Stats Operations in general exclude missing data.

#Performing a descriptive statistic:


# data.mean()

# Same operation on the other axis:

#data.mean(1)
s = pd.Series([1,3,5,np.nan,6,8], index=dates).shift(2)
s
df.sub(s, axis='index')
## Apply
#Applying functions to the data:

data['price'].apply(np.cumsum).head(5)
a = data['points']#.apply(lambda x: x.max() - x.min())

a.max() - a.min()
data['price'].value_counts().head(10)
# String Methods

# Series is equipped with a set of string processing methods in the str attribute that make it easy to operate on each element of the array,
# Note that pattern-matching in str generally uses regular expressions by default (and in some cases always uses them)

strs = data['description']
# lower case

strs.str.lower().head(10)
## Merge

# Concat

#pandas provides various facilities for easily combining together Series, DataFrame, and Panel objects with various kinds of set logic for the indexes and relational algebra functionality in the case of join / merge-type operations.
#Concatenating pandas objects together with concat():

# break it into pieces
pieces = [data[:3], data[3:7], data[7:]]

pd.concat(pieces)
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})

right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
pd.merge(left, right, on='key')
left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
pd.merge(left, right, on='key')
# Append

# Append rows to a dataframe. 

df = pd.DataFrame(np.random.randn(8, 4), columns=['A','B','C','D'])
df
s = data.iloc[3]
s
## Appending

data.append(s, ignore_index=True)
# group by single column

data.groupby('country')['price'].sum().head(10)
# group by multiple columns

data.groupby(['country', 'province'])['price'].sum().head(10)
# Stack

# The stack() method “compresses” a level in the DataFrame’s columns.

stacked = data.stack()
stacked
## Unstack()

stacked.unstack()
## Change in axis

stacked.unstack(0)
# We can produce pivot tables from this data very easily:

pd.pivot_table(data, values='price', index=['country', 'province'], columns=['points'])
#  pandas has simple, powerful, and efficient functionality for performing resampling operations during frequency conversion (e.g., converting secondly data into 5-minutely data).
#  This is extremely common in, but not limited to, financial applications.
rng = pd.date_range('1/1/2012', periods=100, freq='S')
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
ts.resample('5Min').sum()
# Time zone representation:

rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
ts = pd.Series(np.random.randn(len(rng)), rng)
ts
# change time zones

ts_utc = ts.tz_localize('UTC')
ts_utc
# Converting to another time zone:

ts_utc.tz_convert('US/Eastern')
# Converting between time span representations:

rng = pd.date_range('1/1/2012', periods=5, freq='M')

ts = pd.Series(np.random.randn(len(rng)), index=rng)
# Converting between period and timestamp enables some convenient arithmetic functions to be used. In the following example, 
# we convert a quarterly frequency with year ending in November to 9am of the end of the month following the quarter end:

prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')
prng
ts = pd.Series(np.random.randn(len(prng)), prng)
ts
ts.index = (prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9

ts.head()
# Convert the raw grades to a categorical data type.

data["country"] = data["country"].astype("category")
data["country"].head(10)
# Sorting is per order in the categories, not lexical order.

data.sort_values(by="country")
# Grouping by a categorical column also shows empty categories.

data.groupby("country").size()
# Plotting
#ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = data['points']
ts = ts.cumsum()

ts.plot()
df = df.cumsum()

plt.figure(); df.plot(); plt.legend(loc='best')
#CSV

#Writing to a csv file.

df.to_csv('foo.csv')

# Reading

pd.read_csv('foo.csv')

# Writing to a HDF5 Store.

df.to_hdf('foo.h5','df')

# Reading from a HDF5 Store.

pd.read_hdf('foo.h5','df')

# Writing to an excel file.

df.to_excel('foo.xlsx', sheet_name='Sheet1')

# Reading from an excel file

pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])