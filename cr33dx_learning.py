from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

print(os.listdir('../input'))
nRowsRead = None # specify 'None' if want to read whole file

# zomato.csv has 51717 rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('../input/zomato.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'zomato.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
data =df1.copy()
data.loc[:,['address','location','listed_in(city)']].sample(8,random_state=7)
target = ['address','location','listed_in(city)']

for i in target:

    print(f'Number of unique values for {i} :  {data[i].nunique()}')
print(data['listed_in(city)'].unique())
data.head(5)
data.loc[:,'rate'] #index based search 
data[0:4:] #data.head(4)
data.index
data.loc[1] #loc is used to fetch value from index here 1 is first row  data.loc[0:3]
data.columns
data.loc[0:3,['url','name']] #get 4 rows and url,name column we can also do data.loc[0:3,'url':'']
data.iloc[1] == data.loc[1] #both are simmilar but use loc for label based indexing and iloc for index based indexing
data[data.votes == np.nan]
data.iloc[28,:]
data[pd.isna(data.dish_liked) == True] #show all where dish_liked is NaN
data.rate = data.rate.str.split('/',expand = True)[0]

data.rate = data.rate.fillna(1)

data[data.rate == 'NEW'] = 0

data[data.rate == '-'] = 0

data.rate = data.rate.astype('float64')
total_votes = data.rate.multiply(data.votes)

data['total_rating'] = total_votes

location_split = data.groupby('location')

location_split.first() #Print First Entry Of All Of The Groups Formed
rating_location = location_split.total_rating.sum() #use reset_index() to not make location as index

rating_location
rating_location.plot.bar(figsize = (20,10))