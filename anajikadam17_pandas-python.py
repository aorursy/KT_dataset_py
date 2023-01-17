import pandas as pd
#creating a pandas series from a list

import pandas as pd



my_list = [10, 20, 30]

series = pd.Series(my_list)



print(series)

print(series.index)

print(series.values)
# creating a series from numPy Array

import numpy as np

import pandas as pd



index = ['a','b','c']

arr = np.array([10,20,30])



series = pd.Series(data=arr,index=index)

print(series)

print(series.index)

print(series.values)
# creating a series from dictionary

import pandas as pd



d = {'a':10, 'b':20, 'c':30}

pd.Series(d)
# Custom index

import pandas as pd

ser1 = pd.Series([1,2,3,4], index=['USA', 'Germany','USSR', 'Japan']) 

ser2 = pd.Series([1,2,5,4], index=['USA', 'Germany','Italy', 'Japan'])   



# get the value of 'USA'

print(ser1['USA'])
print(ser1 + ser2)
groceries = pd.Series(data = [30, 6, 'Yes', 'No'], index = ['eggs', 'apples', 'milk', 'bread'])

print('Groceries has shape:', groceries.shape)

print('Groceries has dimension:', groceries.ndim)

print('Groceries has a total of', groceries.size, 'elements')

print('The data in Groceries is:', groceries.values)

print('The index of Groceries is:', groceries.index)

print('Groceries:\n', groceries)
# check whether an index label exists in Series

x = 'bananas' in groceries

x
# Accessing Elements

# using index labels:

# single index label

print('How many eggs do we need to buy:', groceries['eggs'])

# access multiple index labels

print('Do we need milk and bread:\n', groceries[['milk', 'bread']]) 

# use loc to access multiple index labels

print('How many eggs and apples do we need to buy:\n', groceries.loc[['eggs', 'apples']]) 



# access elements in Groceries using numerical indices:

# use multiple numerical indices

print('How many eggs and apples do we need to buy:\n',  groceries[[0, 1]]) 

# use a negative numerical index

print('Do we need bread:\n', groceries[[-1]]) 

# use a single numerical index

print('How many eggs do we need to buy:', groceries[0]) 

# use iloc (stands for integer location) to access multiple numerical indices

print('Do we need milk and bread:\n', groceries.iloc[[2, 3]])

# Since we can access elements in various ways, in order to remove

# any ambiguity to whether we are referring to an index label

# or numerical index, Pandas Series have two attributes,

# .loc and .iloc to explicitly state what we mean. The attribute

# .loc stands for location and it is used to explicitly state that

# we are using a labeled index. Similarly, the attribute .iloc stands

# for integer location and it is used to explicitly state that we are

# using a numerical index.
# Change Elements

groceries['eggs'] = 2

groceries
# Delete Elements

# doesn't change the original Series being modified

groceries.drop('apples')

print(groceries)

# delete items from Series in place by setting keyword inplace to True

groceries.drop('apples', inplace = True)

print(groceries)
# Arithmetic Operations

# we can perform element-wise arithmetic operations on Pandas Series

fruits = pd.Series(data = [10, 6, 3,], index = ['apples', 'oranges', 'bananas'])

print(fruits)

print(fruits + 2) # Adds 2 to all elements in the series

print(fruits - 2)

print(fruits * 2)

print(fruits / 2)

# apply mathematical functions from NumPy to all elements of a Series

print(np.exp(fruits))

print(np.sqrt(fruits))

print(np.power(fruits,2))

# only apply arithmetic operations on selected items in Series



print(fruits['bananas'] + 2)



print(fruits.iloc[0] - 2)

print(fruits[['apples', 'oranges']] * 2)

# you can apply arithmetic operations on a Series of mixed data

# type provided that the arithmetic operation is defined for all

# data types in the Series, otherwise you will get an error
import pandas as pd

import numpy as np



df = pd.DataFrame([[1, 2, 3],

                   [3, 4, 5],

                   [5, 6, 7],

                   [7, 8, 9]])

df
df = pd.DataFrame([[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]])



print("Shape:", df.shape)

print("Index:", df.index)



df
df2 = pd.DataFrame([[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]],

                   index=['a', 'b', 'c', 'd'], columns=['x', 'y', 'z'])



print("Shape:", df.shape)

print("Index:", df.index)



df2
df.info()
df.describe()   #describe only for numerical data
# understanding axes

print(df.sum() )      

# sums “down” the 0 axis (rows)

print(df.sum(axis=0) )

# equivalent (since axis=0 is the default)

print(df.sum(axis=1) )

# sums “across” the 1 axis (columns)
df.head(2)
df[1].unique()
df[1].nunique()
df[1].value_counts()
# Create dictionary from a bunch of Series/data

books = pd.Series(data = ['Great Expectations', 'Of Mice and Men', 'Romeo and Juliet', 'The Time Machine', 'Alice in Wonderland' ])

authors = pd.Series(data = ['Charles Dickens', 'John Steinbeck', 'William Shakespeare', ' H. G. Wells', 'Lewis Carroll' ])

user_1 = pd.Series(data = [3.2, np.nan ,2.5])

user_2 = pd.Series(data = [5., 1.3, 4.0, 3.8])

user_3 = pd.Series(data = [2.0, 2.3, np.nan, 4])

user_4 = pd.Series(data = [4, 3.5, 4, 5, 4.2])



# Create a dictionary with the data given above

a_dict = {'Author':authors,'Book Title':books,'User 1':user_1, 'User 2':user_2, 'User 3':user_3, 'User 4':user_4}



# Use the dictionary to create a Pandas DataFrame

book_ratings = pd.DataFrame(a_dict)

book_ratings
# convert to numpy array (remove the column names, get just the values to convert it into a numpy array)

book_ratings_numpy = book_ratings.values

book_ratings_numpy
# Subdataframe

df2 = book_ratings[['Author','Book Title']]

df2
# Create a DataFrame manually from a dictionary of Pandas Series



# create a dictionary of Pandas Series 

items = {'Bob' : pd.Series(data = [245, 25, 55], index = ['bike', 'pants', 'watch']),

         'Alice' : pd.Series(data = [40, 110, 500, 45], index = ['book', 'glasses', 'bike', 'pants'])}



# print the type of items to see that it is a dictionary

print(type(items)) # class 'dict'



# create a Pandas DataFrame by passing it a dictionary of Series

shopping_carts = pd.DataFrame(items)

print(shopping_carts)

# create a DataFrame that only has a subset of the data/columns

bob_shopping_cart = pd.DataFrame(items, columns=['Bob'])



# create a DataFrame that only has selected keys

sel_shopping_cart = pd.DataFrame(items, index = ['pants', 'book'])



# combine both of the above - selected keys for selected columns

alice_sel_shopping_cart = pd.DataFrame(items, index = ['glasses', 'bike'], columns = ['Alice'])



# create DataFrames from a dictionary of lists (arrays)

# In this case, however, all the lists (arrays) in the dictionary must be of the same length



# create a dictionary of lists (arrays)

data = {'Integers' : [1,2,3],

        'Floats' : [4.5, 8.2, 9.6]}



# create a DataFrame 

df = pd.DataFrame(data)



# create a DataFrame and provide the row index

df = pd.DataFrame(data, index = ['label 1', 'label 2', 'label 3'])



# create DataFrames from a list of Python dictionaries

# create a list of Python dictionaries

items2 = [{'bikes': 20, 'pants': 30, 'watches': 35}, 

          {'watches': 10, 'glasses': 50, 'bikes': 15, 'pants':5}]



# create a DataFrame 

store_items = pd.DataFrame(items2)



# create a DataFrame and provide the row index

store_items = pd.DataFrame(items2, index = ['store 1', 'store 2'])



print('shopping_carts has shape:', shopping_carts.shape)

print('shopping_carts has dimension:', shopping_carts.ndim)

print('shopping_carts has a total of:', shopping_carts.size, 'elements')

print()

print('The data in shopping_carts is:\n', shopping_carts.values)

print()

print('The row index in shopping_carts is:', shopping_carts.index)

print()

print('The column index in shopping_carts is:', shopping_carts.columns)
# Access Elements

print()

print('How many bikes are in each store:\n', store_items[['bikes']])

print()

print('How many bikes and pants are in each store:\n', store_items[['bikes', 'pants']])

print()

print('What items are in Store 1:\n', store_items.loc[['store 1']]) #by loc

print()

print('How many bikes are in Store 2:', store_items['bikes']['store 2'])

print('How many bikes are in Store 2:', store_items.iloc[1,0])  #by .iloc
# Rename the row and column labels

# change the column label

store_items = store_items.rename(columns = {'bikes': 'hats'})

# change the row label

store_items = store_items.rename(index = {'store 2': 'last store'})

store_items
# change the index to be one of the columns in the DataFrame

store_items = store_items.set_index('pants')

store_items
# Dealing with NaN values (missing data)



# create a list of Python dictionaries

items2 = [{'bikes': 20, 'pants': 30, 'watches': 35, 'shirts': 15, 'shoes':8, 'suits':45},

{'watches': 10, 'glasses': 50, 'bikes': 15, 'pants':5, 'shirts': 2, 'shoes':5, 'suits':7},

{'bikes': 20, 'pants': 30, 'watches': 35, 'glasses': 4, 'shoes':10}]



# We create a DataFrame and provide the row index

store_items = pd.DataFrame(items2, index = ['store 1', 'store 2', 'store 3'])

print(store_items)

# check if we have any NaN values in our dataset

# .any() performs an or operation. If any of the values along the

# specified axis is True, this will return True.

df.isnull().any()

'''

Date   False

Open   True

High   False

Low    False

Close  False

Volume False

dtype: bool

'''



# count the number of NaN values in DataFrame

x =  store_items.isnull().sum().sum()

print(x)

# count the number of non-NaN values in DataFrame

y = store_items.count()

print(y)



# remove rows or columns from our DataFrame that contain any NaN values

# drop any rows with NaN values

store_items.dropna(axis = 0)



# drop any columns with NaN values

store_items.dropna(axis = 1)



# the original DataFrame is not modified by default

# to remove missing values from original df, use inplace = True

store_items.dropna(axis = 0, inplace = True)



# replace all NaN values with 0

store_items.fillna(0)



# forward filling: replace NaN values with previous values in the df,

# this is known as . When replacing NaN values with forward filling,

# we can use previous values taken from columns or rows.

# replace NaN values with the previous value in the column

store_items.fillna(method = 'ffill', axis = 0)



# backward filling: replace the NaN values with the values that

# go after them in the DataFrame

# replace NaN values with the next value in the row

store_items.fillna(method = 'backfill', axis = 1)



# replace NaN values by using linear interpolation using column values

store_items.interpolate(method = 'linear', axis = 0)



# the original DataFrame is not modified. replace the NaN values

# in place by setting inplace = True inside function

store_items.fillna(method = 'ffill', axis = 0, inplace = True)

store_items.interpolate(method = 'linear', axis = 0, inplace = True)

store_items
df.head()

df.tail()

df.describe()

# prints max value in each column

df.max()



# display the memory usage of a DataFrame

# total usage

df.info()

# usage by column

df.memory_usage()
# get the correlation between different columns

df.corr()
book_ratings.dropna(axis = 1,inplace=True)

book_ratings.head()

# Since we have not mentioned inplace=True, it returns a new dataframe.

book_ratings.drop(labels=['Book Title'], axis=1).head(5)
sorted_df = book_ratings.sort_values('User 4', ascending=False)  # can be inplace as well

sorted_df.head()
sorted_df['User 4'].mean()
data = {

    'A': ['foo','foo','foo','bar','bar','bar'],

    'B': ['one','one','two','two','one','one'],

    'C': ['x','y','x','y','x','y'],

    'D': [1, 3, 2, 5, 4, 1]

}



df = pd.DataFrame(data)

df
pivot_df = df.pivot_table(

                values='D',      # We want to aggregate the values of which column?

                index='A',       # We want to use which column as the new index?

                columns=['C'],   # We want to use the values of which column as the new columns? (optional)

                aggfunc=np.sum)  # What aggregation function to use ?





pivot_df
# convert it back to a simple index



pivot_df.reset_index()
df2 = df.groupby(df['B']).agg(np.mean).reset_index()

df2
df1 = pd.DataFrame({

    'A': ['A0', 'A1', 'A2', 'A3'],

    'B': ['B0', 'B1', 'B2', 'B3'],

    'C': ['C0', 'C1', 'C2', 'C3'],

    'D': ['D0', 'D1', 'D2', 'D3']

}, index=[0, 1, 2, 3])



df2 = pd.DataFrame({

    'A': ['A4', 'A5', 'A6', 'A7'],

    'B': ['B4', 'B5', 'B6', 'B7'],

    'C': ['C4', 'C5', 'C6', 'C7'],

    'D': ['D4', 'D5', 'D6', 'D7']

}, index=[4, 5, 6, 7])



df3 = pd.DataFrame({

    'A': ['A8', 'A9', 'A10', 'A11'],

    'B': ['B8', 'B9', 'B10', 'B11'],

    'C': ['C8', 'C9', 'C10', 'C11'],

    'E': ['D8', 'D9', 'D10', 'D11']

}, index=[8, 9, 10, 11])

df1
pd.concat([df1, df2, df3])
# axis=1 means concat along columns



pd.concat([df1, df2, df3], axis=1)
# Join

left_df = pd.DataFrame({

    'A': ['A0', 'A1', 'A2'],

    'B': ['B0', 'B1', 'B2']

}, index=['K0', 'K1', 'K2']) 



right_df = pd.DataFrame({

    'C': ['C0', 'C2', 'C3'],

    'D': ['D0', 'D2', 'D3']

}, index=['K0', 'K2', 'K3'])



left_df.join(right_df, how='outer')
# Merging on multiple keys

left = pd.DataFrame({

    'key1': ['K0', 'K0', 'K1', 'K2'],

    'key2': ['K0', 'K1', 'K0', 'K1'],

    'A': ['A0', 'A1', 'A2', 'A3'],

    'B': ['B0', 'B1', 'B2', 'B3']

})

    

right = pd.DataFrame({

    'key1': ['K0', 'K1', 'K1', 'K2'],

    'key2': ['K0', 'K0', 'K0', 'K0'],

    'C': ['C0', 'C1', 'C2', 'C3'],

    'D': ['D0', 'D1', 'D2', 'D3']

})



pd.merge(left, right, how='outer', on=['key1', 'key2'])
# other options are 'inner', 'left', 'right'



pd.merge(left, right, how='left', on=['key1', 'key2'])