# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Show the path you will use
data = pd.read_csv('../input/Pokemon.csv')
# info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage
data.info()
data.head()  # Display first five elements of data 
#data.tail() # Display last five elements of data
data.isnull().sum()  # Display the number of null elements
# columns gives column names of features
data.columns
# shape gives number of rows and columns in a tuble
data.shape
# For example, lets take a look at frequency of pokemom types
print(data['Type 1'].value_counts(dropna =False))  # if there are nan values that also be counted
# As it can be seen below there are 112 water pokemon or 70 grass pokemon
# For example, max HP is 255 or min defense is 5
data.describe() # it ignores null entries
# For example: compare attack of pokemons that are legendary  or not
# Black line at top is max
# Blue line at top is 75%
# Green line is median (50%)
# Blue line at bottom is 25%
# Black line at bottom is min
# There are no outliers
data.boxplot(column='Attack', by='Legendary')
# Firstly I create a new data from pokemons data to explain melt more easily.
data_new = data.head()    # I only take 5 rows into new data
data_new
# lets melt it
# id_vars -> what we do not wish to melt
# value_vars -> what we want to melt
melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Attack','Defense'])
melted
# Index is name
# Columns (Attack and Defense) are variable
# And values in columns are value
melted.pivot(index = 'Name', columns = 'variable',values='value')
# Firstly lets create 2 data frames
data1 = data.head()
data2 = data.tail()
# Now lets combine them as vertical (axis=0)
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
conc_data_row
# Now lets combine Attack and Defense columns as horizontal (axis=1)
data1 = data['Attack'].head()
data2 = data['Defense'].head()
conc_data_col = pd.concat([data1,data2],axis =1)
conc_data_col
data.dtypes  # Display data types of columns
# lets convert object(string) to categorical and integer to float.
data['Type 1'] = data['Type 1'].astype('category')
data['Speed'] = data['Speed'].astype('float')
# Let's see differences
# As you can see Type 1 is converted from object to categorical
# And Speed is converted from integer to float
data.dtypes
# Lets look at does pokemon data have nan value
# As you can see there are 800 entries. However Type 2 has 414 non-null object so it has 386 null object.
data.info()
# Lets check Type 2
data["Type 2"].value_counts(dropna =False)
# As you can see, there are 386 NAN values
# Lets drop nan values
data1=data   # also we will use data to fill missing value so I assign it to data1 variable
data1["Type 2"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data
# So does it work ?
# Let's check with assert statement
# Assert statement:
assert 1==1 # return nothing because it is true
# In order to run all code, we need to make this line comment
# assert 1==2  # returns an error because it is false
assert data['Type 2'].notnull().all() # returns nothing because we drop nan values
data["Type 2"].fillna('empty',inplace = True)
assert data['Type 2'].notnull().all() # returns nothing because we do not have nan values
# With assert statement we can check lots of things. For example;
# assert data.columns[1] == 'Name'
# assert data.Speed.dtypes == np.int