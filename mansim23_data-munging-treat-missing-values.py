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
#By default, missing numbers are represented by NaN i.e. Not a Number. One common way of treating missing values is by replacing them with an average of the values available. It is better than dropping the missing values altogether.
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
#Use a variable calling 'missing' to indicate missing values using the nan function
missing=np.nan
series_obj=Series(['row1', 'row2', missing, 'row4', 'row5', 'row6', missing, 'row8'])
series_obj
#Finding a missing values using the isnull() method
series_obj.isnull()
#Filling the missing values
np.random.seed(25)
df_obj=DataFrame(np.random.randn(36).reshape(6,6))
df_obj
df_obj.ix[3:5,0]=missing
df_obj.ix[1:4,5]=missing
df_obj
#Fill an NaN value with 0 using the fillna() function and store this new matrix in another variable called filled_df
filled_df=df_obj.fillna(0)
filled_df
df_obj
#You can use dictionary to replace a value in a certain cell with something else. This is done through the dict parameter in the fillna() function. 
#Inside the dictionary param, the value before the : is the column and the value after the : is the value to replace it with
filled_df=df_obj.fillna({0:0.1,5:1.25})
filled_df
#We can use the fill-forward feature using the ffill method to fill an empty value with the last non-null value of that column
df_filled_new=df_obj.fillna(method='ffill')
df_filled_new
#Counting missing values
#This is useful esp when you want to know which columns in your data set are the most problematic
#We use the isnull() to return the missing values and then apply the sum() to count the true values that are missing. The values are always returned column wise.
df_obj.isnull().sum()
#We might want to drop all the rows that contain missing values. This can be done using the dropna() function
df_noNaN=df_obj.dropna()
df_noNaN
#Sometimes, you might want to drop the column which contains blank values and not the row. This can be done using the axis parameter. By default, the parameter is set to 0 i.e. row. When axis-1, it drops the column and prints all other cols with row
df_noNaN=df_obj.dropna(axis=1)
df_noNaN
#Sometimes, you only want to drop those rows which contain all missing values. For this, we use the how='all' inside the dropna() function
df_noNaN=df_obj.dropna(how='all')
df_noNaN
