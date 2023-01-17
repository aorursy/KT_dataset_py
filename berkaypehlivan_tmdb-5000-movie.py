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
import matplotlib.pyplot as plt
movie_data = pd.read_csv('../input/tmdb_5000_movies.csv')
movie_data.info() #Returns the basic informations about the data
movie_data.columns 
movie_data.corr() #Gives the relation between columns
movie_data.head() #Returns the first 5 movies on the list
movie_data.tail() #Returns the last 5 movies on the list
print(movie_data[['budget']])  #You can also use  print(movie_data.budget)
print(movie_data.loc[:4,'budget':"id"]) #Gives first 4 rows from budget to id
print(movie_data.loc[:4,['budget','id']]) #Gives first 4 rows but only 'budeget' and 'id' rows
print(movie_data.loc[::-1,:]) #Returns reversed list of the movies
filter1 = movie_data.original_language == 'en'
print(movie_data[filter1])
filter2 = movie_data.runtime > 90.0
print(movie_data[filter2])
print(movie_data[filter1&filter2]) 
print(movie_data[np.logical_and(movie_data.vote_average > 7.0,movie_data.runtime <120.0)]) #Instead of '&' np.logicial_and can be used 
average_runtime = movie_data.runtime.mean()
movie_data['runtime_level'] = ['Long' if each > average_runtime else 'Short' for each in movie_data.runtime ]
print(movie_data.loc[:10,:'runtime_level']) #Runtime level for first 10 movies
average_popularity = movie_data.popularity.mean()
movie_data['popularity_level'] = ['High Popularity' if each > average_popularity else 'Low Popularity' for each in movie_data.popularity ]
print(movie_data.loc[:10,:'popularity_level']) #Popularity level of first 10 movies
#Concatenating data

data1 = movie_data.budget.head()

data2 = movie_data.budget.tail()



data_v_concat= pd.concat([data1,data2],axis = 0)

data_h_concat = pd. concat([data1,data2],axis =1)

print(data_v_concat)

print(' ')

print(data_h_concat)
dictionary = {'id': '19995','genre': 'Action','name': 'Avatar'}

print(dictionary.keys())

print(dictionary.values())
dictionary['id'] = '285' # update existing entry

print(dictionary)

dictionary['year'] = 2009 # Add new entry

print(dictionary)

del dictionary['id']  # remove entry with key 'id'

print(dictionary)

print('year' in dictionary) # check include or not

dictionary.clear()   # remove all entries in dictionary

print(dictionary)
# Stays in loop if condition( x is lower than 10) is true

x = 0

while x< 10:

    print('x is',x)

    x = x+1

print(x,'is equal to 10')
# Stay in loop if condition( i is not equal 5) is true

list1 = [1,2,3,4,5]

for i in list1:

    print('i is: ',i)

print('')

#Returns sum of the list2 values

list2 = [3,5,7,9,11,13,15,17,19]

count = 0

for each in list2:

    count = count+each

print(count)

print('')

# Enumerate index and value of list

# index : value = 0:1, 1:2, 2:3, 3:4, 4:5

for index,value in enumerate(list1):

    print(index,':',value)

print('')  

# For dictionaries

# We can use for loop to achive key and value of dictionary.

dictionary2 = {'id': '185', 'genre': 'Action', 'name': 'Lord of The Rings', 'year': 2001}

for key,value in dictionary2.items():

    print(key," : ",value)

print('')

# For pandas we can achieve index and value

for index,value in movie_data[['budget']][0:2].iterrows():

     print(index," : ",value)

def f (x,y=2,z=5): # y and z is default arguments

    s = x+y+z

    return(s)

print(f(10))



def f(*args):  # flexible arguments *args

    for i in args:

        print(i)

f(5,6,7,8)



def f(**kwargs):  #print key and value of dictionary

    for key,value in kwargs.items():

        print(key,' ',value)

f( ıd = '19995',genre = 'Action',name = 'Avatar')
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line



movie_data.budget.plot(kind = 'line',color = 'r',label = 'budget',linewidth =1,alpha = 0.5,grid = True,linestyle = ':',figsize = (10,10))

movie_data.popularity.plot(kind = 'line',color ='g',label = 'popularity',linewidth = 1,alpha = 0.5,grid = True,linestyle = '-.',figsize = (10,10))

plt.legend(loc = 'upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
#Scatter Plot

# x = budget y = revenue

movie_data.plot(kind = 'scatter',x = 'budget',y = 'revenue',alpha = 0.5,color = 'blue',figsize = (10,10))

plt.xlabel('Budget')

plt.ylabel('Revenue')

plt.title('Budget Revenue Scatter Plot')
# Histogram

# bins = number of bar in figure



movie_data.vote_average.plot(kind = 'hist',bins = 40,figsize = (10,10))

plt.show()