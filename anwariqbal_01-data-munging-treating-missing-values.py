import numpy as np
import pandas as pd 

from pandas import Series, DataFrame
missing = np.nan

series_obj = Series(['row 1', 'row 2', missing, 'row 4','row 5', 'row 6', missing, 'row 8'])
series_obj
# object_name.isnull()
# The .isnull() method returns a Boolean value that describes (True or False) whether an element in a 
# Pandas object is a null value.
series_obj.isnull()
np.random.seed(25)
DF_obj = DataFrame(np.random.randn(36).reshape(6,6))
DF_obj
DF_obj.ix[3:5, 0] = missing
DF_obj.ix[1:4, 5] = missing
DF_obj
# object_name.fillna(numeric value)
# The .fillna method() finds each missing value from within a Pandas object and fills it with the 
# numeric value that you've passed in.
filled_DF = DF_obj.fillna(0)
filled_DF
# object_name.fillna(dict)
# You can pass a dictionary into the .fillna() method. The method will then fill in missing values 
# from each column Series (as designated by the dictionary key) with its own unique value 
# (as specified in the corresponding dictionary value).
filled_DF = DF_obj.fillna({0: 0.1, 5: 1.25})
filled_DF
# You can also pass in the method='ffill' arguement, and the .fillna() method will fill-forward any 
# missing values with values from the last non-null element in the column Series.
fill_DF = DF_obj.fillna(method='ffill')
fill_DF
np.random.seed(25)
DF_obj = DataFrame(np.random.randn(36).reshape(6,6))
DF_obj.ix[3:5, 0] = missing
DF_obj.ix[1:4, 5] = missing
DF_obj
# object_name.isnull().sum()
# To generate a count of how many missing values a DataFrame has per column, just call the .isnull() 
# method off of the object, and then call the .sum() method off of the matrix of Boolean values it 
# returns.
DF_obj.isnull().sum()
# object_name.dropna()
# To identify and drop all rows from a DataFrame that contain ANY missing values, simply call the 
# .dropna() method off of the DataFrame object. NOTE: If you wanted to drop columns that contain 
# any missing values, you'd just pass in the axis=1 argument to select and search the DataFrame 
# by columns, instead of by row.
DF_no_NaN = DF_obj.dropna(axis=1)
DF_no_NaN
# object_name.dropna(how='all')
# To identify and drop only the rows from a DataFrame that contain ALL missing values, simply 
# call the .dropna() method off of the DataFrame object, and pass in the how='all' argument.
DF_obj.dropna(how='all')
