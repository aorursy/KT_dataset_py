# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
s = pd.Series([1,3,5, np.nan, 6,8])

s
dates = pd.date_range('2013', periods=6)

dates
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))

df
data_set = pd.read_csv("../input/00-fruits.csv")

data_set
# values in the form of an Array

data_set.values 
# skip the last column 

data_set[data_set.columns[:-1]] 
data_set.index
data_set.columns
data_set['Label']
data_set.dtypes
data_set.shape
data_set.head()
data_set.describe()
data_set.values
s=pd.Series(['a','a', 'b', 'c'])

s.describe()
# class distribution : # of instances for each class

# here, Label is the column name that defines the class/category

data_set.groupby('Label').size()
count_classes = pd.value_counts(data_set['Label'], sort = True).sort_index()

count_classes
data_set.values # 2-D array
data_set.values.flatten() # 1-D array
data_set.values.flatten()[0] # 1-D array
data_set.values[0][:-1] # skip the last value
# skip the last column 

data_set[data_set.columns[:-1]] 