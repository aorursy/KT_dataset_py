import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.



import os

print(os.listdir("../input"))
data = pd.read_csv("../input/countries of the world.csv")
data.info() # shows some information about the dataset.
data.head() # shows first 5 rows.
data.tail() # shows last 5 rows.
data.shape # shows # of samples and # of features/columns.
print(data['Region'].value_counts(dropna=True)) #dropna defines don't include Not A Number(NaN) values.

#Here we ignore them.
data.describe()
data.boxplot(column = 'GDP ($ per capita)', figsize =(28,15), by = 'Region') # column is showed data feature.

# Datas received from by = '....'
data_new = data.head()

data_new
melted = pd.melt(frame = data_new, id_vars = 'Country', value_vars = ['Population', 'GDP ($ per capita)','Birthrate'])

melted
melted.pivot(index = 'Country', columns = 'variable', values = 'value')
data1 = data.head()

data2 = data.tail()



data_conc_row = pd.concat([data1,data2], axis = 0, ignore_index = True) # axis = 0 concatenates dataframes in row.

data_conc_row
data1 = data['Population'].head()

data2 = data['Birthrate'].head()



data_conc_column = pd.concat([data1,data2], axis = 1) # axis = 1 concatenates dataframes in column.

data_conc_column 
data.dtypes
data['Country'] = data['Country'].astype('category') # Data type of the Country feature is changed object to category

data['Population'] = data['Population'].astype('float') # Data type of the Population feature is changed integer to float
data.dtypes
data['Country'] = data['Country'].astype('object') # Converted default type

data['Population'] = data['Population'].astype('int') # Converted default type
data.dtypes
data['Literacy (%)'].value_counts(dropna = False) # dropna = False shows frequency of NaN
data1 = data

data1['Literacy (%)'].dropna(inplace = True)
assert 1==1 # returns nothing because it's true
assert data1['Literacy (%)'].notnull().all() # we drop NaN values, so it returns nothing
data1['Literacy (%)'].fillna('empty', inplace = True)
assert data1['Literacy (%)'].notnull().all() # we do not have NaN values,thus returns nothing