import numpy as np

import pandas as pd

from pandas import Series, DataFrame
data = {

            'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],

            'year': [2000, 2001, 2002, 2001, 2002],

            'pop': [1.5, 1.7, 3.6, 2.4, 2.9]

       }

frame = DataFrame(data)

frame
frame2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'], index=['one', 'two', 'three', 'four', 'five'])

frame2
obj = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])

obj2 = Series([8, 1, 2, -21], index=['d', 'b', 'a', 'c'])

DataFrame({'col1': obj, 'col2': obj2})
frame3 = DataFrame({'year': {'one': 2000, 'two': 2001, 'three': 2002, 'four': 2001, 'five': 2002}, 

                    'state': {'one': 'Ohio', 'two': 'Ohio', 'three': 'Ohio', 'four': 'Nevada', 'five': 'Nevada'},

                    'pop': {'one': 1.5, 'two': 1.7, 'three': 3.6, 'four': 2.4, 'five': 2.9}})

frame3
frame2.values
frame2.shape
frame2.index
frame2.columns
frame2['state'] 
frame2.state
frame2[['state', 'debt']] 
frame2['one':'three']  
frame2[:5]  
frame2.reindex(['two', 'one']) 
frame2.loc['one':'three', ['state', 'debt']]  
frame2.loc[:, ['state', 'debt']] # All the rows with selected columns
frame2.loc['one':'one', ['state', 'debt']]    # one row with selected columns
frame2.loc['one', ['state', 'debt']]    # one row, get a series
frame2.iloc[0:3, [1,3]]    # iloc is counting index not the column/row names - recommended
frame2.iloc[:, [1,3]]    # all the rows
frame2.iloc[0:1, [1, 3]]    # one row, get DataFrame
frame2.iloc[0, [1, 3]]    # one row, get Series
frame2['debt'] = 16.5

frame2
frame2['debt'] = np.arange(5)

frame2
val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])

frame2['debt'] = val

frame2
frame2.name = 'data frame'

frame2.columns.name = 'columns'

frame2.index.name = 'index'

frame2
frame2.columns = ['a', 'b', 'c', 'd']

frame2
frame2.index = [1, 2, 3, 4, 5]

frame2
frame2.drop([1, 2], inplace=True)    # Delete rows, inplace=True means it will delete in the original table

frame2
frame2.drop(['a', 'b'], axis=1)    # Delete columns

frame2
frame2.pop('c')    

frame2
del frame2['a']

frame2
frame3['pop'] > 2
frame3[frame3['pop'] > 2]
frame['pop'].apply(lambda x: x*3)   