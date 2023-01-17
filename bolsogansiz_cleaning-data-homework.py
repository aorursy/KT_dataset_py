# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/pokemon/Pokemon.csv')
data.head()
data.tail()
data.info()
data.columns
data.shape
#For example lets look frequency of pokemom types

print(data['Type 1'].value_counts(dropna =False))  # if there are nan values that also be counted
data.describe()
# Compare attack of pokemons that are legendary  or not

# Black line at top is max

# Blue line at top is 75%

# Red line is median (50%)

# Blue line at bottom is 25%

# Black line at bottom is min

# There are no outliers

data.boxplot(column='Attack',by = 'Legendary')
data_new = data.head()
data_new
# lets melt

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Attack','Defense'])

melted
# Pivoting Data (Reverse Melting)

# Index is name

# I want to make that columns are variable

# Finally values in columns are value

melted.pivot(index = 'Name', columns = 'variable',values='value')
data1 = data.head()

data2 = data.tail()

data_conc_row = pd.concat([data1,data2],axis = 0,ignore_index = True) #axis = 0 : adds dataframes in row

# if we don't make True ignore_index, then indexes of data2 would start at 719 in this example

data_conc_row
data1 = data['Attack'].head()

data2= data['Defense'].head()

data_conc_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row

data_conc_col
data.dtypes
# lets convert object(str) to categorical and int to float.

data['Type 1'] = data['Type 1'].astype('category')

data['Speed'] = data['Speed'].astype('float')
data.dtypes
# Lets look at does pokemon data have nan value

# As you can see there are 800 entries. However Type 2 has 414 non-null object so it has 386 null object.

data.info()
# Lets check Type 2

data["Type 2"].value_counts(dropna =False)

# As you can see, there are 386 NAN value
# Lets drop nan values

data1=data.copy()   # also we will use data to fill missing value so I assign it to data1 variable

data1["Type 2"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data

# So does it work ?
#  Lets check with assert statement

# Assert statement:

assert 1==1 # return nothing because it is true
# In order to run all code, we need to make this line comment

assert 1==2 # return error because it is false
assert data1['Type 2'].notnull().all() # returns nothing because we drop nan values
data["Type 2"].fillna('empty',inplace = True)
data.head()  # we can see Charmander's Type 2 feature is not 'NaN' it is 'empty'
assert  data['Type 2'].notnull().all() # returns nothing because we do not have nan values
# # With assert statement we can check a lot of thing. For example

assert data.columns[1] == 'Name' # it returns nothing, because our index 1 column is 'Name'
assert data.Speed.dtypes == np.int # it returns AssertionError beacuse we have changed the datatype of Speed column to float from int