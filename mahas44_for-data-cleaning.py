# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Pokemon.csv")
data.head()
data.tail()
data.columns
data.shape
data.info()
# For example lets look frequnecy of pokemon types

print(data["Type 1"].value_counts(dropna = False)) # if there are nan values that also be counted
# For example max HP is 255 or min defense is 5

data.describe() # ignore null entries
# circles mean outlies value



data.boxplot(column='Attack', by = 'Legendary')

plt.show()
# Firstly I create new data from pokemons data to explain melt nore easily

data_new = data.head()

data_new
# lets melt

# id_vars = what we do not wish to melt

# value_vars = what we watn to melt

melted = pd.melt(frame=data_new, id_vars = 'Name', value_vars = ['Attack','Defense'])

melted
# PIVOTING DATA

# Reverse of melting



# Index is name

# I want to make that columns are variable

# Finally values in columns are value



melted.pivot(index = 'Name', columns = 'variable', values = 'value')
# Firstly lets create 2 data frame



data1 = data.head()

data2 = data.tail()

conc_data_row = pd.concat([data1,data2], axis = 0, ignore_index = True) # axis = 0 : adds dataframes in row

conc_data_row
data3 = data['Attack'].head()

data4 = data['Defense'].head()

conc_data_col = pd.concat([data3,data4], axis = 1) # axis = 1 : adds dataframes in column

conc_data_col
data.dtypes
# Lets convert object(str) to categorical and it to float

data['Type 1']= data['Type 1'].astype('category')

data['Speed'] = data['Speed'].astype('float')

data.dtypes
data.info()

# Type 2 has 414 non-null object so it has 386 null object
data['Type 2'].value_counts(dropna = False)
data1 = data

data1['Type 2'].dropna(inplace = True)
assert data1['Type 2'].notnull().all # returns nothing because we drop nan values
data['Type 2'].fillna('empty',inplace = True)
assert data['Type 2'].notnull().all() # returns nothing because we don't have nan values