# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np 

import pandas as pd 
animals = pd.Series(['dog','cat','cow','lion','tiger'])

animals
# A default index is created since we did not specify explicitly. 

animals.index
animals.values
# We can specify the index while creating the series object

animals2 = pd.Series(['dog','cat','cow','lion','tiger'], index=['a','b','c','d','e'])

animals2
# Both the series object and the index can be assigned a name.

animals2.name = 'animals'

animals2.index.name = 'order'

animals2
# Using index to select values 

print('Result of a single index input: \n {}'.format(animals2['a']))

print('Result of a single index input: \n {}'.format(animals2[['b', 'c', 'd']])) # interpreted as list of indices
# Create series from a dict.

data = {'Russia':'Moscow', 'India': 'New Delhi', 'USA':'DC', 'China':'Beijing'}

countries = pd.Series(data)

countries
# Change the order of the keys in the series.

key_order = ['Russia', 'China', 'USA', 'India']

countries = pd.Series(data, index=key_order)

countries
# Instance methods

animals2.isnull()
animals2.notnull()
# Creating a df from a dict of equal length lists

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],

        'year': [2000, 2001, 2002, 2001, 2002, 2003],

        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}

df = pd.DataFrame(data)

df # Observe that the index is assigned automatically.
# We can also pass column names & index list while creating the data frame object.

df = pd.DataFrame(data, columns=['year','state','pop','debt'], index=['one','two','three','four','five','six'])



# Observe that debt is assigned missing values since our dict does not have it as keys.

df 
# Retrieving a column as a series either by dict-like notation or by attribute



print("Retrieve by dict notation: \n {}".format(df['state']))

print("Retrieve by attribute: \n {}".format(df.state))



# Note

# df[column] works for any column name, but df.column only works when the column name is a valid Python variable name.
# Modifying columns by assignment 

df['debt'] = 10

df
# Length of the array should match the length of the index.

df['debt'] = np.arange(6)

df
# If we assign a series with an index to a column in a data frame, the index of the series will me matched with the data frame

# and the values will be assigned accordingly. 



val = pd.Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])

df['debt'] = val

df

df['eastern'] = (df.state == 'Ohio')

df
# Deleting a column in data frame

del df['eastern']

df
# Data Frame from Nested dictionaries

# Outer keys are taken as columns and inner keys as row indices.



pop = {'Nevada': {2001: 2.4, 2002: 2.9},

        'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}



df3 = pd.DataFrame(pop)

df3
# Setting the index and column name attributes on a data frame

df3.index.name = 'year'

df3.columns.name = 'state'

df3
# Data as 2-D array

df.values
obj = pd.Series(range(3), index=['a', 'b', 'c'])

idx = obj.index

idx
# Index objects are immutable

# idx[2] = 'd' # Error



# It is safe to share index objects among data structures since they are immutable.

labels = pd.Index(['one','two','three'])

print(labels)

# We will use labels as index for another series

obj2 = pd.Series([1.3, 3.4, 5.6], index=labels)

print(obj2)
# Check if the index of object 2 is labels

obj2.index is labels
obj = pd.Series([4.5, 6, 7.4, 5], index=['d','b','a','c'])

obj
obj2 = obj.reindex(['a', 'b', 'c', 'd'])

obj2
# With DataFrame, reindex can alter either the (row) index, columns, or both. 

# When passed only a sequence, it reindexes the rows in the result.



df = pd.DataFrame(np.arange(9).reshape((3, 3)),

                  index=['a', 'c', 'd'],

                  columns=['Ohio', 'Texas', 'California'])

df
df2 = df.reindex(index=['a', 'b','c', 'd'])

df2
# Reindexing columns

states = ['Texas', 'Utah', 'California']

df2.reindex(columns=states)
# Reindexing columns with a fill_value

states = ['Texas', 'Utah', 'California']

df2.reindex(columns=states, fill_value = 0)
obj = pd.Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])

obj
obj2 = obj.drop('c')

obj2
obj.drop(['c', 'd'])
# Deleting index values froma data frame.

data = pd.DataFrame(np.arange(16).reshape((4, 4)),

                     index=['Ohio', 'Colorado', 'Utah', 'New York'],

                     columns=['one', 'two', 'three', 'four'])

data

data.drop(['Colorado', 'Ohio'])
data.drop(['two', 'four'], axis=1)
data.drop(['two', 'four'], axis='columns')
obj = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])

obj
obj['b']
obj[1]
obj[['b', 'a', 'd']]
obj[[1, 3]]
obj[obj<2]
obj['b':'d']
df = pd.DataFrame(np.arange(16).reshape(4,4), 

                  columns=['Russia', 'India', 'US', 'China'], 

                  index=['one', 'two', 'three', 'four'])

df
df['Russia']
df[['Russia', 'India']]
# Selecting the first two rows

df[:2]
df<5
df[df<5] = 0
df
ds = pd.DataFrame(np.arange(16).reshape(4,4), 

                  columns=['Canada', 'India', 'Brazil', 'UK'], 

                  index=['one', 'two', 'three', 'four'])

ds
# Select single column from df.

ds['Canada']



# Select a sequence of columns

ds[['Canada', 'India']]
# Selects single row or subset of rows from the DataFrame by label.

ds.loc['one']
ds.loc[['one', 'three']]
# Selects single column or subset of columns by label.

ds.loc[:, 'Canada']
ds.loc[:, ['Canada', 'UK']]
# Selecy both rows and columns by label

ds.loc['one','Canada']
# Select single row or subset of rows by integer position

ds.iloc[[2,3]]
# Select single column or subset of columns by integer position

ds.iloc[:, [2,3]]
# Select both rows and columns by integer position

ds.iloc[[1,2], [2,3]]
# Interger indexes

s = pd.Series(np.arange(3))

s
s[1]
# Throws an error if we have an integer based index.

# When the index is integer based, position based indexing throws an error. 

s[-1]
df = pd.DataFrame(np.random.randint(40, 55, 9).reshape(3,3), 

                  columns=["Sachin", "Ponting", "Lara"], 

                  index=(["Test", "ODI", "T20"]))

df
np.ceil(df)
# Applying a function to each column or row.

f = lambda x:x.max() - x.min()



# Invokes the function once per column

df.apply(f, axis = 0) # is same as df.apply(f, axis = 'rows') 
# Invokes the function once per row

df.apply(f, axis = 1) # is same as df.apply(f, axis = 'columns') 
# Function need not return a scalar value 

# For each player in the dataframe return min and max value. 

# Which means for each column return the min and max value



def f(x):

    return pd.Series([x.min(), x.max()], index=['min_val', 'max_val'])



df.apply(f, axis=0)

# Apply a function to each element of a data frame.

g = lambda x : (1.2*x)

df.applymap(g)
# Use only when a vectorised version of the function does not exist as its faster than applymap.

df * 1.2
# Sorting by row or column index.

s = pd.Series(np.arange(4), index=['a','d','c', 'b'])

s
s.sort_index()
# Sort by values in the series

s.sort_values(ascending=False)
# DataFrame can be sorted on either indices.

df.sort_index(axis=1, ascending=False) 
# Sort DataFrame by values

df = pd.DataFrame({'a': np.random.randn(5), 'b': np.random.randint(0,34,5)})

df
df.sort_values(by='a')
df.sort_values(by=['a', 'b'])
# Ranking 

# By default rank breaks ties by assigning each group the mean rank.



s = pd.Series([1,1,2,3,4,5,5,6,7,8,8,9])

s.rank()
# Various options for tie breaking

s.rank(method='first')

s.rank(method='average')

s.rank(method='dense')

s.rank(method='min')

s.rank(method='max')
# Ranking for DataFrame

df = pd.DataFrame({'a': np.random.random(5), 'b': np.random.random(5), 'c': np.random.randint(10,30,5)})

df
df.rank(axis=0)
data = pd.Series(np.random.randn(9),

                  index=[

                          ['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],

                          [1, 2, 3, 1, 3, 1, 2, 2, 3]

                  ])

data
data.index
# Partial Indexing

data['a']
# Inner level partial indexing

data.loc[:,[2,3]]
data.unstack()
# Incase of a data frame either index can have hierarchical index.



frame = pd.DataFrame(np.arange(12).reshape((4, 3)),

                      index=[

                          ['a', 'a', 'b', 'b'], 

                          [1, 2, 1, 2]

                      ],

                      columns=[

                          ['Ohio', 'Ohio', 'Colorado'],

                          ['Green', 'Red', 'Green']

                      ])
frame
# The hierarchical index can have names as well.

frame.index.names = ['key1', 'key2']

frame.columns.names = ['state', 'color']
frame
# Partial column indexing.

frame['Colorado']
frame.swaplevel('key1', 'key2')
df1 = pd.DataFrame({

    'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],

    'data1': range(7)

})



df2 = pd.DataFrame({

    'key': ['a', 'b', 'd'],

    'data2': range(3)})
df1
df2
# Many to one join 

pd.merge(df1, df2)
pd.merge(df2, df1)
# Merge uses the overlapping column names as the keys. It’s a good practice to specify explicitly.



pd.merge(df1, df2, on='key')
# If column names are different on each object then we have to specify them separately.


