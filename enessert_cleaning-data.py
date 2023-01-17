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
data =pd.read_csv('../input/Pokemon.csv')     # data read
data.info()     #data's information
data.head()  # first 5 rows
data.tail()  # last 5 rows
data.columns # give me column names
data.shape  ## shape gives number of rows and columns in a tuble
# frequency of pokemon types
print(data['Type 2'].value_counts(dropna = False))   # if there are nan values that also be counted
# For example max Speed is 180 or min attack is 5
data.describe() # ignore null entries
# For example: compare defense of pokemons that are legendary  or not
# Black line at top is max
# Blue line at top is 75%
# Green line is median (50%)
# Blue line at bottom is 25%
# Black line at bottom is min
# There are no outliers
# -------------------------------------------
# boxplot parameters
# column : Column name or list of names, or vector.
# by : Column in the DataFrame to pandas.DataFrame.groupby(). 
# ax : The matplotlib axes to be used by boxplot.
# fontsize : Tick label font size in points or as a string (e.g., large).
# grid : Setting this to True will show the grid.
# figsize : The size of the figure to create in matplotlib.

data.boxplot(column='Defense',by = 'Legendary',fontsize = 'large', figsize = (8,8) )
# Merge data
new_data = data.head()  # I only take 5 rows into new data
new_data   #show new_data

# lets melt
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted = pd.melt(frame=new_data,id_vars ='Name' , value_vars =['HP','Speed'])
melted     # show melted
# Lets reverse
melted.pivot(index ='Name' , columns ='variable', values = 'value' )
# Create 2 dataframe
data1 = data.head()
data2 = data.tail()
conc_data_row = pd.concat([data1,data2],axis=0, ignore_index=True ) # axis=0 dataframes in row 
conc_data_row   # show
# Create 2 dataframe
data1 = data['HP'].head()
data2 = data['Speed'].head()
conc_data_col = pd.concat([data1,data2],axis=1 )  
conc_data_col   # show
data.dtypes
# convert object(str) -----> categorical
# convert int ------> float
data['Type 1'] =data['Type 1'].astype('category')
data['Defense'] =data['Defense'].astype('float')
# Type 2 changed from object to category
# Speed changed from int to float
data.dtypes
#Type 2 has 414 non-null object so it has 386 null object
data.info()
#Lets chech Type 2
data["Type 2"].value_counts(dropna =False)
#Lets drop nan values (delete)
data1 = data   # also we will use data to fill missing value so I assign it to data1 variable
data1['Type 2'].dropna(inplace = True) # inplace = True means we do not assign it to new variable. Changes automatically assigned to data
# Assert statement:
assert 1==1   # return nothing because it is true
# False so give me error
#assert 1==2
assert data['Type 2'].notnull().all()  # returns nothing because we drop nan values
data["Type 2"].fillna('empty',inplace=True)
assert data['Type 2'].notnull().all()  # returns nothing because we do not have nan values