import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np
#Creating a Series

s = pd.Series([12,-4,7,9])

print(s)

# By default pandas will create a level from 0

# This can be changed

s = pd.Series([12,-4,7,9], index=['a','b','c','d'])

# Accessing the index and value of a series

s.values

s.index
# Selecting element of a series

s[2] #can be done by positional index

# Can be done by label as well

s['b']

# Multiple Elements can also be selected from continious position

s[0:2]

# Multiple Elements can also be selected from discontinious position

z=np.array([1,3])

s[z]

s[['a','c']]
s[1] = 5

print(s)

s[s > 8]

s==5

s[s==5]
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
# Selecting Rows and Columns

df[1:2]

df['price']

# Selecting third and fifth row

df.iloc[[2,4]]

# Selecting third and fifth row, second and third column

df.iloc[[2,4],[1,2]]

df['new']=[12,13,14,14,16]

df

# Deleting column

df.drop(['object'],axis=1)
# reading from file

df = pd.read_csv('../input/btissue/btissue.csv')

# read_csv('ch05_02.csv',skiprows=[2],nrows=3,header=None)

# Similarly read_excel, read_json, read_html etc. is available

# Read_table can be used with text files and separators can be user defined

# Examine first few rows

df.head(3)

# Check the name of the columns

print(df.columns)

print(df.shape)

print(df.dtypes)

print(df.info)

# Getting the values of IO when class = car

df['I0'][df['class']=='car']



df.iloc[:,[1,2]][df['class']=='car']

# Getting the vale of IO when class = car

df['I0'][df['class']=='car'].mean()





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