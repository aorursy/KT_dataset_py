import numpy as np

import pandas as pd



from pandas import Series, DataFrame
series_obj = Series(np.arange(8), index=['row 1', 'row 2','row 3','row 4','row 5', 'row 6', 'row 7', 'row 8'])

series_obj
# ['label-index'] When you write square brackets with a label-index inside them, 

# This tells Python to select and retrieve all records with that label-index.

series_obj['row 7']
# [integer index] When you write square brackets with an integer index inside them, 

# This tells Python to select and retrieve all records with the specified integer index.

series_obj[[0,7]]
np.random.seed(50)

DF_obj = DataFrame(np.random.rand(36).reshape((6,6)), 

                   index=['row 1', 'row 2', 'row 3', 'row 4', 'row 5', 'row 6'],

                   columns=['column 1', 'column 2', 'column 3', 'column 4', 'column 5', 'column 6'])

DF_obj
# object_name.ix[[row indexes], [column indexes]]

# When you call the .ix[] special indexer, and pass in a set of row and colum indexes, 

# This tells Python to select and retrieve only those specific rows and columns.



DF_obj.ix[['row 2', 'row 5'], ['column 5', 'column 2']]
# ['starting label-index':'ending label-index'] 

# Data slicing allows you to select and retrieve all records from the starting label-index, to the 

# ending label-index, and every record in between.



series_obj['row 3':'row 7']
# object_name < scalar value

# You can use comparison operators (like greater than or less than) to return True / False values for 

# all records, to indicate how each element compares to a scalar value. 

DF_obj < .2
# object_name[object_name > scalar value] 

# You can also use comparison operators and scalar values for indexing, to return only the records 

# that satisfy the comparison expression you write.



series_obj[series_obj > 6]
# ['label-index', 'label-index', 'label-index'] = scalar value

# Setting is where you select all records associated with the specified label-indexes and set those 

# values equal to a scalar.

series_obj['row 1', 'row 5', 'row 8'] = 8
series_obj