import numpy as np

import pandas as pd



from sklearn import datasets
# Loads a kiiind of dictionary/object with attributes 

# referring to info about the dataset.

dataset = datasets.load_boston() 
print(dataset.DESCR)
data = pd.DataFrame(columns = dataset.feature_names, data = dataset.data)



# Full docs for DataFrame objects is here:

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html



# .head() gives the first few items from the dataset 

# so you can get a better look at the layout and features 

# (referred to as variables in Intro to Statistics) of the dataset.



data.head()
# Access different features/columns like a dictionary.



# Each column is a Series object, basically a normal list where 

# all elements have the same type (str, float, int, etc.) with 

# some new helpful methods and functions thrown in, feel free 

# to read the names of the methods. For most methods reading 

# the name is enough. It's pretty easy to guess that the .sum() 

# method of a Series of numbers does.



# Full Series docs here:

# http://pandas.pydata.org/pandas-docs/stable/reference/series.html



data['AGE'].head()
# To get specific elements of a DataFrame or Series using their positions use the 'iloc' attribute.



data['AGE'].iloc[50]



# data.iloc[50] will obviously also work