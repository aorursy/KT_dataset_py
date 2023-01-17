# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/turkish-airlines-daily-stock-prices-since-2013/cleanThy.csv')
#information of data

data.info()

#the ration between data

data.corr()
#correlation map #

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
#top 10 data but ----- data.head() = top 5 data

data.head(10)
#last 10 data

data.tail()
data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data[' Lowest Price'].plot(kind = 'line', color = 'r',label = 'Lowest Price',linewidth=1,alpha = 0.5,grid = True,linestyle = ':',figsize = (15,15))

data[' Highest Price'].plot(kind = "line",color = 'b',label = 'Highest Price',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.',figsize = (15,15))

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Price')            # title = title of plot

plt.show()
# Scatter Plot =  correlation between two variables

# x = Lowest Price , y = Highest Price

data.plot(kind = 'scatter' , x=' Lowest Price', y=' Highest Price' , alpha = 0.5 , color = 'red', figsize = (12,12))

plt.xlabel('Lowest')

plt.ylabel('Highest')

plt.title('Price')

plt.show()
#Alternative Code

plt.scatter(data[' Lowest Price'],data[' Highest Price'],alpha = 0.5,color='red')
# Histogram

# bins = number of bar in figure

data[' Last Price'].plot(kind = 'hist',bins = 50 , figsize = (12,12))

plt.show()
# clf() = cleans it up again you can start a fresh

data[' Last Price'].plot(kind = 'hist',bins = 50)

plt.clf()

# We cannot see plot due to clf()
#create dictionary and look its keys and values

dic = {'Lowest Price' : '13.02','Highest Price' : '13.28'}

print(dic.values())

print(dic.keys())
# Keys have to be immutable objects like string, boolean, float, integer or tubles

# List is not immutable

# Keys are unique

dic['Lowest Price'] = '13.10'  #update existing entry

print(dic)

dic['Last Price'] = '13.25'  #add new entry

print(dic)

del dic['Lowest Price']        #remove entry with key 'Lowest Price'

print(dic)

print('Last Price' in dic)

dic.clear()                   # remove all entries in dict.Delete entire dictionary from memories so it gives error because dictionary is deleted

print(dic)
data = pd.read_csv('../input/turkish-airlines-daily-stock-prices-since-2013/cleanThy.csv')
# 1 - Filtering Pandas data frame

filter1 = data[' Lowest Price']>10

data[filter1]
# 2 - Filtering pandas - use '&' for filtering.

data[(data[' Lowest Price']>10) & (data[' Highest Price']<15)]

# Stay in loop if condition( i is not equal 5) is true

i = 0

while i < 5 :

    print('i is: ',i)

    i = i + 1

print(i,' is not less than 5')
# Stay in loop if condition( i is not equal 5) is true

liste = [12,13,14,15]

for i in liste:

    print('Price :',i )

    

# Enumerate index and value of list

# index : value = 0:1, 1:2, 2:3, 3:4, 4:5

for index,value in enumerate(liste):

    print(index ,":", value)

    

# For dictionaries

# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.

dic={'First Price ':'12','Second Price:':'13'}

for key,value in dic.items():

    print(key,":",value)
# For pandas we can achieve index and value

for index,value in data[[' Lowest Price']][0:4].iterrows():

    print(index," : ",value)
# example of what we learn above

def tuble_ex():

    """ return defined t tuble"""

    t = (11,12,13)

    return t

Lowest,Highest,Last = tuble_ex()

print(Lowest,Highest,Last)
# guess print what

x = 12

def f():

    x = 13

    return x

print(x)      # x = 12 global scope

print(f())    # x = 13 local scope
# What if there is no local scope

x = 10

def f():

    y = 4*x        # there is no local scope x

    return(y)      # it uses global scope x

print(f())    

# First local scopesearched, then global scope searched, if two of them cannot be found lastly built in scope searched.    
# How can we learn what is built in scope

import builtins

dir(builtins)
#nested function



def Square():

    """return square of value"""

    def add():

        """add three local variable"""

        x = 5

        y = 6

        z = 8

        w = x+y+z

        return w 

    return add()**2

print(Square())
#default arguments



def f(x , y =  5 , z = 10):

    w = x + y + z

    return w

print(f(20))

# what if we want to change default arguments

print(f(7,8,9))

    
#flexible arguments *args



def f(*args):

    for i in args:

        return i

print(f('Lowest , Highest , Last'))



# flexible arguments **kwargs that is dictionary

def f(**kwargs):

    for key,value in kwargs.items():

        print(key, ':' ,value)

    

f(country = 'France', capital = 'Monaco', population = 123456)    

    
#lambda function

square = lambda x : x**2    # where x is name of argument

print(square(5))

total = lambda x,y,z : x+y+z    # where x,y,z are names of arguments

print(total(5,6,7))
lis1 = [1,2,3]

z = map(lambda x : x**2,lis1)

print(z)

print(list(z))
#iteration example

name = "Monaco"

it = iter(name)

print(next(it))

print(*it)

#zip example

lis1 = [1,2,3]

lis2 = [4,5,6]

z = zip(lis1,lis2)

print(z)

z_list = list(z)

print(z_list)
#example of list comprehension

lis1 = [10,15,20]

lis2 = [i + 5 for i in lis1]

print(lis2)
# Conditionals on iterable

lis1 = [5,10,15]

lis2 = [i + 5 if i>10 else i + 10 if i == 10 else i-5 for i in lis1]

print(lis2)
#comprehension example

#Our treshold is Highest Price

treshold = sum(data[' Last Price'])/len(' Last Price')

data['price_level'] = ['High' if i > treshold else 'Low' for i in data[' Last Price']]

data.loc[:10,['price_level',' Last Price']]

data = pd.read_csv('../input/turkish-airlines-daily-stock-prices-since-2013/cleanThy.csv')
data.head()  # head shows first 5 rows
# tail shows last 5 rows

data.tail()
# columns gives column names of features

data.columns
# shape gives number of rows and columns in a tuble

data.shape
#information of data

data.info()
#for example

print(data[' Volume'].value_counts(dropna = True)) # if there are nan values that also be counted
data.describe() #ignore null entries  #%25 Q1 quartile -- %50 Q2 median----%75 Q3 qurtile
# Black line at top is max

# Blue line at top is 75%

# Red line is median (50%)

# Blue line at bottom is 25%

# Black line at bottom is min

# There are no outliers

data.boxplot(column = ' Highest Price',by = 'price_level')
#Firstly I create new data

data_new = data.head()

data_new
# lets melt

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted = pd.melt(frame = data_new,id_vars = 'Date',value_vars = [' Lowest Price',' Highest Price'])

melted                                                                    
#Index is Date

# I want to make that columns are variable

# Finally values in columns are value

melted.pivot(index = 'Date',columns = 'variable',values = 'value')
#Firstly I creat two data frame

data1 = data.head()

data2 = data.tail()

concat_data_row = pd.concat([data1,data2],axis=0) #axis = 0 in row

concat_data_row
data1 = data.head()

data2 = data.tail()

concat_data_column = pd.concat([data1,data2],axis=1) 

concat_data_column
# lets convert object(str) to categorical and int to float.

data[' Lowest Price'] = data[' Lowest Price'].astype('category')

data[' Highest Price'] = data[' Highest Price'].astype('float')
data.dtypes
# Lets look at does THY Price data have nan value

data.info()
#Lets check type 2

data[' Highest Price'].value_counts(dropna = False)

#As you can see,there is no NaN value
#Lets assume that our data have NaN value

#Lets drop NaN value

data[' Highest Price'].dropna(inplace = 1)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data

data[' Highest Price'].value_counts(dropna = False)

#  Lets check with assert statement

# Assert statement:

assert 1==1 # return nothing because it is true
# In order to run all code, we need to make this line comment

# assert 1==2 # return error because it is false
assert  data[' Highest Price'].notnull().all() # returns nothing because we drop nan values
# replacing NaN values in data with empty 

data[' Highest Price'].fillna('empty',inplace = True)

assert  data[' Highest Price'].notnull().all() # returns nothing because we drop nan values
# # With assert statement we can check a lot of thing. For example

assert data.columns[1] == ' Highest Price'

#returns error because first column's name is 'Date'