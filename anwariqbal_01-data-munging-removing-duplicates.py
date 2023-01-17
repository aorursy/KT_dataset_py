import numpy as np
import pandas as pd

from pandas import Series, DataFrame
DF_obj = DataFrame({'column 1': [1, 1, 2, 2, 3, 3, 3],
                  'column 2': ['a', 'a', 'b', 'b', 'c', 'c', 'c'],
                  'column 3': ['A', 'A', 'B', 'B', 'C', 'C', 'C']})
DF_obj
# object_name.duplicated()
# The .duplicated() method searches each row in the DataFrame, and returns a True or False value to 
#indicate whether it is a duplicate of another row found earlier in the DataFrame.
DF_obj.duplicated()
# object_name.drop_duplicates()
# To drop all duplicate rows, just call the drop_duplicates() method off of the DataFrame.
DF_obj.drop_duplicates()
DF_obj = DataFrame({'column 1': [1, 1, 2, 2, 3, 3, 3],
                  'column 2': ['a', 'a', 'b', 'b', 'c', 'c', 'c'],
                  'column 3': ['A', 'A', 'B', 'B', 'C', 'D', 'C']})
DF_obj
# object_name.drop_duplicates(['column_name'])
# To drop the rows that have duplicates in only one column Series, just call the drop_duplicates() 
# method off of the DataFrame, and pass in the label-index of the column you want the de-duplication 
# to be based on. This method will drops all rows that have duplicates in the column you specify.
DF_obj.drop_duplicates(['column 3'])
