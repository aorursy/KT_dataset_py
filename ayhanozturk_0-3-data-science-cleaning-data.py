import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

data = pd.read_csv('../input/pokemon.csv')
data.head(6)
data.tail(6)
data.columns
data.shape
data.info()
data.describe()
#############
print(data['Type 2'].value_counts(dropna=False))
data.boxplot(column='Attack')
data.boxplot(column='Attack',by = 'Legendary')
data_new=data.head(100).loc[20:25,["Type 1","Type 2","HP","Attack"]]

data_new
melting_data=pd.melt(frame=data_new,id_vars='Type 1',value_vars=['Type 2','HP','Attack'])

melting_data

data_new = data.head()    # I only take 5 rows into new data

data_new

melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Attack','Defense'])

melted
melted.pivot(index = 'Name', columns = 'variable',values='value')
data1 = data['Type 1'].head()

data2= data['Type 1'].tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row
data.dtypes
data['Type 1']=data['Type 1'].astype('category')

data['Speed'] = data['Speed'].astype('float')
data.dtypes
data.info() #Type 2 414 not equal to 800
data['Type 2'].value_counts(dropna=False)
data1=data.copy()
data1["Type 2"].dropna(inplace = True)
assert data1['Type 2'].notnull().all()
data['Type 2'].fillna('empty',inplace=True)
data1.info()

data1['Type 2'].value_counts(dropna=False)
assert  data1['Type 2'].notnull().all()