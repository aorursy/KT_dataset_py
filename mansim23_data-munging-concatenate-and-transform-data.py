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
#Concatenation is useful in cases where multiple data frames can be combined into a single dataset
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
df_obj=DataFrame(np.arange(36).reshape(6,6))
df_obj
df_obj_2=DataFrame(np.arange(15).reshape(5,3))
df_obj_2
#The concat() method joins on the rows when axis=1 making the table wider. If there is no axis specified, it concats on the column
df_concat=pd.concat([df_obj,df_obj_2],axis=1)
df_concat
df_concatcols=pd.concat([df_obj,df_obj_2])
df_concatcols
#The drop() funtion is used to drop a row by default. If we want to drop a column, then axis=1
df_droprows=df_obj.drop([0,2])
df_droprows
df_dropcols=df_obj.drop([0,2],axis=1)
df_dropcols
#We can add data to a dataframe by creating a series and then adding it to a df
series_add=Series(np.arange(6))
series_add.name='added_variable'
series_add
#Use the .join method to add a Series to a Dataframe
df_joined=DataFrame.join(df_obj, series_add)
df_joined
#We can use the .append method to add dataframes below as rows
added_datatable=df_joined.append(df_joined,ignore_index=True)
added_datatable
#We can sort data using the .sort_values function. It takes 2 params: first is the index of the column by which we want to sort, second is Asending=True/False
df_obj.sort_values(by=2,ascending=[False])
