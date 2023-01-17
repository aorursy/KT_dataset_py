# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/tmdb_5000_movies.csv')
data.info() # we get info about our dataframe
data.corr() # shows the correlation amongst the integer and float datas
# correlation map
f, ax = plt.subplots(figsize =(16,16))
sns.heatmap(data.corr(), annot = True, linewidths = 0.5, fmt = ".1f", ax=ax) 
# fmt .1 shows the digits that will be written after the dot
plt.show()

data.head(10) 
# shows the first 10 rows to get info about the big picture
# the default value shows 5 rows
data.columns 
# shows the column names
# Line Plot

data.budget.plot(kind = 'line', color = 'red', label = "Budget", linewidth = 1, alpha = .8, grid = True, linestyle = "-")
data.popularity.plot(kind = 'line', color = 'blue', label = "Popularity", linewidth = 1, alpha = .7, grid = True, linestyle = ":")

plt.legend(loc='upper right')
plt.xlabel('Budget')
plt.ylabel('Popularity')
plt.title('TMDB Movies Budget-Popularity')
plt.show()
# Scatter Plot

data.plot(kind = 'scatter', x='budget', y='popularity', alpha=0.3, color= "green")
plt.xlabel('Budget')
plt.ylabel('Popularity')
plt.title('TMDB Movies Budget-Popularity')

# Histogram

data.popularity.plot(kind='hist', bins=100,figsize=(16,16), color='purple')
plt.show()
# Dictionary

dict_01 = {'country_abv':'US', 'country_name' : 'United States of America'}
print(dict_01.keys())
print(dict_01.values())
dict_01['company']='Walt Disney' # adding new entry in dictionary
print(dict_01)
print('UK'in dict_01)     # checks whether 'UK' exists in dictionary
# Pandas

data = pd.read_csv('../input/tmdb_5000_movies.csv')
series = data['popularity']
print(type(series))
data_frame = data[['popularity']]
print(type(data_frame))
# Pandas_01 Filtering Pandas Data Frame

# To find out movies with popularity over 300 and budget under 150000000

x = data['popularity']>300
y = data['budget']<150000000
data[x&y]
# To find out Japanesse movies with an vote average over 8.0

a = data['vote_average']>8
b = data['original_language']=='ja'
data[a&b]

# Pandas_01 Filtering Pandas With logical_and

# To find out Japanesse movies with popularity over 100

data[np.logical_and(data['original_language']=='ja',data['popularity']>100)]

# This code gives the same output as the one in line 53

data[(data['original_language']=='ja') & (data['popularity']>100)]
# While and For Loops

dict_01 = {'country_abv':'US', 'country_name' : 'United States of America'}
for key,value in dict_01.items():
    print(key," : ",value)
print('')
