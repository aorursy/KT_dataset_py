# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns # visualize



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/creditcard.csv")
data.info()
data.columns
data.corr()
fig,ax = plt.subplots(figsize =(15,15))

sns.heatmap(data.corr(), annot=True, linewidths=5, fmt='.1f', ax=ax)

plt.show()
data.head(10)
plt.scatter(data.V1, data.V2, alpha=0.5, color='red')

plt.show()
data.plot(kind='scatter', x='V1', y='V2', alpha=0.5, color='green')

plt.xlabel('V1')

plt.ylabel('V2')

plt.title('V1 & V2 Scatter Plot')

plt.show()
data.V1.plot(kind='hist', bins=70, figsize=(15,15))

plt.show()
data.V1.plot(kind = 'hist', bins=50)

plt.clf()
my_dict = {'DE':'deutchland', 'USA':'United States', 'TR':'turkey'}

print(my_dict.keys())

print(my_dict.values())
my_dict['DE'] = 'DEUCTHLAND' # update

print(my_dict)



my_dict['FR'] = 'France' # Add new entry

print(my_dict)

my_dict['SP'] = 'Spain'  # Add new entry

print(my_dict)



del my_dict['USA']

print(my_dict)



my_dict.clear()

print(my_dict)
# del my_dict

print(my_dict)
# Önemli !!!!

# Series x Data Frame



my_series = data['V1']  # Pandas Data Type - Series

print(type(my_series))

my_data_frame = data[['V1']] # Pandas Data Type - Data Frame

print(type(my_data_frame))
# Data Filtering

x = data ['V1'] > 2.42

data[x]
data[(data['V1']>2.42) & (data['V2']> -0.9)]
i = 0

while i != 10 :

    print('i is = ',i)

    i +=2

print(i, ' is equal to 10')
# For loops

lis = [1,2,3,4]



for i in lis:

    print('i is : ', i)

print('')



for index, values in enumerate(lis): # use enumerate for list's index

    print(index, " : ", values)

print('')



for k,v in my_dict.items():

    print(k, " : ", v)

print('')



for index, value in data[['V1']][0:1].iterrows():

    print(index, " : ", value)

print('')
# User Defined Functions, 

# Buil in scope : print, len etc.

# docstring -- """ string for function """



import builtins

dir(builtins)
# Nested Functions

def square():

    """ return square of value """

    def add():

        """ add two local variable """

        x = 2

        y = 3

        z = x+y

        return z

    return add()**2

print(square())
# Default and Flexible Arguments

def f(a, b=1, c =2):

    y = a + b +c

    return y

print(f(5))

print(f(5,4,3)) # default values updated
# flexible arguments

def f(*args):

    for i in args:

        print(i)

f(1)

f(1,2,3,4)
# flexible arguments for Dictionary

def f(**kwargs):

    """ print key and value of dictionary """

    for key, value in kwargs.items():

        print(key, " ", value)

f(country = 'spain', capital='madrid', population = 123456)
# use defined function (long way)

def square(x):

    return x**2

print(square(5))



# lamda function (short way)

square = lambda x: x**2 # where x is name of argument

print(square(4))



tot = lambda x,y,z: x+y+z

print(tot(1,2,3))
# Anonymous Function

# map(function, sequence)



number_list = [1,2,3]

y = map( lambda x:x**2, number_list)

print(list(y))
# Iterations

# iteration example

name = "ronaldo"

it = iter(name)

print(next(it)) # print next iteration

print(*it)      # print remaining iteration
# zip function

list1 = [1,2,3,4]

list2 = [5,6,7,8]

z = zip(list1, list2)

print(z)



z_list = list(z)

print(z_list)
un_zip = zip(*z_list)

un_list1, un_list2 = list(un_zip) # unzip returns tuple

print(un_list1)

print(un_list2)

print(type(un_list2))

print(type(list(un_list1)))
# Important.. !

num1 = [1,2,3]  # num1 iterable object

num2 = [i+1 for i in num1] # list comprehension

print(num2)
# Conditionals on iterable

num1 = [5,10,15]

num2 = [i**2 if i == 10 else i-2 if i < 7 else i+5 for i in num1]

print(num2)