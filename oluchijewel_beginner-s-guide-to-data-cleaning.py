import numpy as np

import pandas as pd

from numpy import NaN
df = pd.DataFrame(np.random.randn(4, 6))
df.head()
df.iloc[:1, 1] = NaN

df.iloc[:3, 2] = NaN
df.head()
df.isnull()
df.dropna()
#Passing how='all' will only drop rows that are all NaN:
df.dropna(how='all')
df[6] = NaN   #adding a new column
df.head()
#To drop columns in the same way, pass axis=1:
df.dropna(axis=1, how='all')
df.fillna(0)
#Calling fillna with a dict, you can use a different fill value for each column:

df.fillna({1: 0.5, 2: 1.0, 6:-2.0})
#fillna returns a new object, but you can modify the existing object in-place:
df.fillna(0, inplace=True)
df.head()
#The same interpolation methods available for reindexing can be used with fillna:

df.fillna(method='ffill')
#Using a new series

df1 =pd.DataFrame([[1., 6.5, 3.], [1., NaN, NaN], [NaN, NaN, NaN], [NaN, 6.5, 3.],  [6, 4, 4],]) 

df1
# You might pass the mean or median value of a Series

df1.fillna(df1.mean())
df1.fillna(0, inplace=True)     #inplace Modify the calling object without producing a copy
df1.head()