# importing useful libraries

import pandas as pd # data processing

import numpy as np

import os
mySeries = pd.Series([3, -5, 7, 4], index = ['a', 'b', 'c', 'd'])

print(mySeries)

print(type(mySeries))
data = {'Country' : ['Belgium', 'India', 'Brazil'],

       'Capital' : ['Brussels', 'New Delhi', 'Brassilia'],

       'Population' : [12345,  123456, 98745]}



df = pd.DataFrame(data, columns = ['Country', 'Capital', 'Population'])

print(df)

print(type(data))

print(type(df))
# Let's make a dataframe of 5 columns and 20 rows

pd.DataFrame(np.random.rand(20, 5))