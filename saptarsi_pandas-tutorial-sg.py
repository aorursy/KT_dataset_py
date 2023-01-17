import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

# The Series is the object of the pandas library designed to represent one-dimensional data structures

# There is a value and an index

s = pd.Series([12,-4,7,9])

print(s)

# By default pandas will create a level from 0

# This can be changed

s = pd.Series([12,-4,7,9], index=['a','b','c','d'])

# Accessing the index and value of a series

s.values

s.index
# Selecting element of a series

s[2] #can be done by index

# Can be done by label as well

s['b']

# Multiple Elements can also be selected

s[0:2]

s[['b','c']]
# Assigning and Filtering Values

s[1] = 5

s[s > 8]

s[s==5]

s==5
# Operations and Mathematical functions

s / 2

np.log(s)
# Taking look at values

serd = pd.Series([1,0,2,1,2,3], index=['white','white','blue','green','green','yellow'])

serd.unique()

serd.value_counts()

# isin( ) is a function that evaluates the membership, that is, given a list of values, this function

#lets you know if these values are contained within the data structure.

serd.isin([0,3])
# Some members can be Null or Not a number

s2 = pd.Series([5,-3,np.NaN,14])

s2.isnull()

s2.notnull()

s2[s2.notnull()]
# Series as Dictionaries

mydict = {'red': 2000, 'blue': 1000, 'yellow': 500, 'orange': 1000}

myseries = pd.Series(mydict)
# The DataFrame is a tabular data structure very similar to the Spreadsheet

data = {'color' : ['blue','green','yellow','red','white'],

'object' : ['ball','pen','pencil','paper','mug'],

'price' : [1.2,1.0,0.6,0.9,1.7]}

df = pd.DataFrame(data)

df

# The number of columns and order can be changed

df2 = pd.DataFrame(data, columns=['object','price'])

df2

# Labels can be assigned similar to series

df3= pd.DataFrame(data, index=['one','two','three','four','five'])

df3
# Selecting Rows and Columns

df3[1:2]

df3['price']

# Selecting third and fifth row

df3.iloc[[2,4]]

# Selecting third and fifth row, second and third column

df3.iloc[[2,4],[1,2]]

# Selecting all odd rows

df.iloc[lambda x: x.index % 2 != 0]

# Adding new colum

df3['new']=[12,13,14,14,16]

df3

# Deleting column

df2.drop(['object'])
# reading from file

df = pd.read_csv('../input/btissue.csv')

# read_csv('ch05_02.csv',skiprows=[2],nrows=3,header=None)

# Similarly read_excel, read_json, read_html etc. is available

# Read_table can be used with text files and separators can be user defined

# Examine first few rows

df.head(3)

# for Writing to csv

df.to_csv()

# parameter na_rep ='NaN'

df.columns

df.shape

df.dtypes

df.info



# Getting the values of IO when class = car

df['I0'][df['class']=='car']



# Getting the vale of IO when class = car

df['I0'][df['class']=='car'].mean()

# Merging dataframes

frame1 = pd.DataFrame( {'id':['ball','pencil','pen','mug','ashtray'],'price': [12.33,11.44,33.21,13.23,33.62]})

frame2 = pd.DataFrame( {'id':['pencil','pencil','ball','pen'],'color': ['white','red','red','black']})

pd.merge(frame1,frame2)

# As the name of the column on the basis of which the merging will happen has same names, it doe snot

# need to be specified, otherwise it can be added with the ON parameter

pd.merge(frame1,frame2,on='id')

# by default merge is inner join, if we need to add other joins we can specify the 'how' parameter



# Assignment create two dataframe one has studendid and marks and another has student id and phone number

# The first dataframe will have values like s1,s2,s3 and 75,78,82 the second dataframe

# will have values like s1,s2,s3 and phone number like 9998764523 etc, Merge them