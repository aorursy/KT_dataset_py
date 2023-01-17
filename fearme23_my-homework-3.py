# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/anime.csv")
data.head() # head show first 5 row
data.tail() # tail show last 5 row
# columns gives column names of features
data.columns
# shape gives number of rows and columns in a tuble
data.shape
# info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage
data.info()
# For example lets look frequency of anime genre
data.genre.value_counts(dropna=False) # if there are nan values that also be counted
data.describe() #ignore null entries
# This codes skipping
filter1 = data.type == "Movie"
filter2 = data.type == "TV"
data1 = data[filter1]
data2 = data [filter2]
ver_concatdata =pd.concat([data1,data2],axis=0,ignore_index=True) # axis = 0 vertial concatenating
ver_concatdata
hor_concatdata = pd.concat([data1,data2],axis = 1) # axis = 1 horizontal concatenating
hor_concatdata
# For example: compare attack of pokemons that are legendary  or not
# Black line at top is max
# Blue line at top is 75%
# Red line is median (50%)
# Blue line at bottom is 25%
# Black line at bottom is min
# There are no outliers
cdata.boxplot(column='members',by ="type",figsize = (24,12))
# Firstly I create new data from anime data to explain melt nore easily.
data_new = data.head(10)   # I only take 10 rows into new data
data_new
# lets melt
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted = pd.melt(frame=data_new,id_vars = 'name', value_vars= ['type','episodes'])
melted
melted.pivot(index = "name",columns="variable",values="value")
data.dtypes


# lets convert object(str) to categorical and int to float.
data.type = data.type.astype('category')
data.members = data.members.astype("float")

# As you can see Type 1 is converted from object to categorical
# And Speed ,s converted from int to float
data.dtypes


# Lets look at does pokemon data have nan value
# As you can see there are 800 entries. However Type 2 has 414 non-null object so it has 386 null object.
data.info()


# Lets chech Type 2
data.genre.value_counts(dropna =False)
# As you can see, there are 62 NAN value


# Lets drop nan values
datax=data   # also we will use data to fill missing value so I assign it to data1 variable
datax.genre.dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data
# So does it work ?
#  Lets check with assert statement
# Assert statement:
assert 1==1 # return nothing because it is true
assert  datax.genre.notnull().all() # returns nothing because we drop nan values


datax.genre.fillna('empty',inplace = True)


data.type.value_counts(dropna=False)
#data.type.fillna("empty",inplace = True)# give error. Because category feature is not working this metod
data.type = data.type.astype("object")# well we convert category to object
data.type.fillna('empty',inplace = True)# 
data.type.value_counts(dropna=False)
assert  data.type.notnull().all()# returns nothing because we do not have nan values