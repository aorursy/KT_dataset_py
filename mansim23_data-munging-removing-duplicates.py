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
#Removing duplicates is important else the same row can be counted multiple times and skew the results. Eg. while analyzing shopping patterns, the same user using 3 credit cards will be treated as 3 people skewing analysis.
import numpy as np
import pandas as pd
from pandas import Series, DataFrame 
df_obj=DataFrame({'col1':[1,1,2,2,3,3,3],'col2':['a','a','b','b','c','c','c'],'col3':['A','A','B','B','C','C','D']})
df_obj
#The function duplicated() returns true if the record is duplicate
df_obj.duplicated()
#Use the function drop_duplicates() to remove the duplicates from a dataframe. 
df_dropped=df_obj.drop_duplicates()
df_dropped
#The drop_duplicates function can also take a column name as a parameter and it drops all the rows for which a duplicate is found in that particular column
db_drop_cols=df_obj.drop_duplicates(['col3'])
db_drop_cols
