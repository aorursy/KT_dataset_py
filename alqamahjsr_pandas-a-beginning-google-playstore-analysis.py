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
#Reading the CSV file ( googleplaystore dataset ) into a DataFrame.

df1 = pd.read_csv("../input/googleplaystore.csv")
#Representing the dimensionality of the DataFrame.

#Tuple represents the number of records and fields in our dataset.

df1.shape
#Returns the column names.

df1.columns
#Show first 5 records of the dataset.

#Specifying n values returns first n rows.

df1.head()
#Show last 5 records of the dataset.

#Specifying n values returns last n rows.

df1.tail()

#Returns summary of a DataFrame including the index dtype and column dtypes, non-null values and memory usage.

df1.info()
#Returns unique values across a series.

df1.Category.unique()

#Returns the maximum of the values in the object.

df1.Rating.max()
#Returns the maximum of the values in the object.

df1.Rating.min()
#Returns data type of each field in the dataset.

df1.dtypes
#Return the sum of the values.

df1.sum()
#Groups series of columns using a mapper

df2 = df1.groupby(['Category']).count()

print(df2)