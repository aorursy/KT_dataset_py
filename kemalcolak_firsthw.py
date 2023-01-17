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

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/master.csv')



print(data)



data.info()



data.corr()



data.head(10)

data.columns



# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.suicides_no.plot(kind = 'line', color = 'g',label = 'suicides_no',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.population.plot(color = 'r',label = 'population',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = attack, y = defense

data.plot(kind='scatter', x='suicides_no', y='gdp_per_capita ($)',alpha = 0.5,color = 'red')

plt.xlabel('suicides_no')              # label = name of label

plt.ylabel('gdp_per_capita ($)')

plt.title('GDP per capita and suicides no')            # title = title of plot
# Histogram

# bins = number of bar in figure

data.year.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
# clf() = cleans it up again you can start a fresh

data.year.plot(kind = 'hist',bins = 50)

plt.clf()

# We cannot see plot due to clf()
#create dictionary and look its keys and values

dictionary = {'spain' : 'madrid','usa' : 'vegas'}

print(dictionary.keys())

print(dictionary.values())
dictionary['spain'] = "barcelona"

print(dictionary)

dictionary['france'] = "paris"

del dictionary ['spain']

print('france' in dictionary) 

dictionary.clear()                   # remove all entries in dict

print(dictionary)

series = data['suicides_no']        # data['suicides_no'] = series

print(type(series))

data_frame = data[['suicides_no']]  # data[['suicides_no']] = data frame

print(type(data_frame))
# Comparison operator

print(3 > 2)

print(3!=2)

# Boolean operators

print(True and False)

print(True or False)
data[data['suicides_no']==data['suicides_no'].max()]
# 1 - Filtering Pandas data frame

x = data['suicides_no']>21000 # There are only 4 records that has more then 21000 suicides

data[x]



# 2 - Filtering pandas with logical_and

# There are only 2 records

data[np.logical_and(data['suicides_no']>21000 , data['population']>19249600 )]

# Stay in loop if condition( i is not equal 5) is true

i = 0

while i != 5 :

    print('i is: ',i)

    i +=1 

print(i,' is equal to 5')
# Stay in loop if condition( i is not equal 5) is true

lis = [1,2,3,4,5]

for i in lis:

    print('i is: ',i)

print('')



# Enumerate index and value of list

# index : value = 0:1, 1:2, 2:3, 3:4, 4:5

for index, value in enumerate(lis):

    print(index," : ",value)

print('')   



# For dictionaries

# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.

dictionary = {'spain':'madrid','france':'paris'}

for key,value in dictionary.items():

    print(key," : ",value)

print('')



# For pandas we can achieve index and value

for index,value in data[['year']][0:1].iterrows():

    print(index," : ",value)
