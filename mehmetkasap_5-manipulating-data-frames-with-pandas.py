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
data = pd.read_csv('../input/pokemon.csv')
data.head()      
# Indexing data frames
data = data.set_index('#') # here # is a feature
data.head()
#%%               
# indexing using square brackets
data.Name[5] # 5 th row of our data
# indexing using loc function 
data.loc[5,:] # 5 th row 
data.loc[5, ['Name', 'Type 1', 'Type 2']]
# selecting only some columns
data['Name'] # only names
data[['Name', 'Type 1', 'Type 2']] # only name, type1 and type 2
# Slicing and indexing data frame
data.loc[1:10, 'HP':'Defense'] 
# reverse slicing data frame
data.loc[10:1:-1, 'HP':'Defense']
# slicing from something to the end
data.loc[793: , 'Speed':]
# filtering data frames
data[data.HP>200]
# using 2 filter
filter1 = data.HP>150
filter2 = data.Speed>40
filter1_and_2 = np.logical_and(filter1,filter2)
data [filter1_and_2]
# or we can use:
data[filter1 & filter2]
# filtering column based others
data.HP[data.Attack >= 180]
# Transforming data using apply function
def multiplyby2(n):
    return 2*n

data.HP.apply(multiplyby2)
# Transforming data using apply and lambda function 
data.HP.apply(lambda n: n*2 )  
# define a new column using existing columns
data ['total_power'] = data.Attack + data.Defense + data.Speed
# INDEX OBJECTS AND LABELED DATA

# print index name
print(data.index.name)
# change index name as number
data.index.name = 'number'
print(data.index.name)
data.head()
# create copy of our data
data_copy = data.copy()
# change index
data_copy.index = range(100,900,1)
data_copy.head()
# lets rewrite our first data 
data = pd.read_csv('../input/pokemon.csv')
data.head()
# lets set 2 indexes: Type 1 outer index and Type 2 inner index
data1 = data.set_index(['Type 1', 'Type 2'])
data1.head(20)
# using new indexes and loc function
data1.loc['Fire', 'Flying'] # first index=Fire second index=Flying
data1.loc['Fire']
# PIVOTING DATA FRAMES
# lets first create a dictionary and a data frame
dic = {'treatment': ['A', 'A', 'B', 'B'],
       'gender': ['F', 'M', 'F', 'M'],
       'response': [10, 45, 5, 9],
       'age': [15, 4, 72, 65]}
df = pd.DataFrame(dic)
df
# Pivoting: Reshape tool
df.pivot(index='treatment', columns='gender', values='response')
# STACKING AND UNSTACKING DATA FRAME
# deal with multi label indexes
# level: position of unstacked index
# swaplevel: change inner and outer level index position
df1 = df.set_index(['treatment', 'gender'])
# unstack outer index
df1.unstack(level=0)
# unstack inner index
df1.unstack(level=1)
# swaplevel: change levels of inner and outer indexes
df2 = df1.swaplevel(0,1)
df2
# MELTING DATA FRAMES
pd.melt(df, id_vars='treatment', value_vars=['response', 'age'])
# CATEGORICALS AND GROUPBY
# groupby: groups our data 
# using mean
df.groupby('treatment').mean() # mean is aggregation or reduction 
# using max (or other methods: sum, min, std)
df.groupby('gender').max()
# after grouping data we can select only one feature
df.groupby('gender').age.mean()
# after grouping data we can select also multiple feature
df.groupby('gender')['age', 'response'].mean()