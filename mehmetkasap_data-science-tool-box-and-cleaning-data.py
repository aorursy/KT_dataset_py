# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/football.csv')
# there is space in some of the column names e.g. Secondary Skill
# and this causes us problems in reaching to that column using DataFrames
# for example the 'data.Secondary Skill' syntax will gives us errors
# to solve this problem:
#data.columns = [each.split()[0]+'_'+each.split()[1] if (len(each.split())>1) else each for each in data.columns]

# lets see the difference by running the following code
data.columns
def tuble_ex(t):
    ''' return a list starting from 1 and until t'''
    liste = []
    i=0
    while i<t:
        i+=1
        liste.append(i)
        
    return liste

my_list = tuble_ex(5)

a, b, c, d = tuble_ex(4)
print(a,b,c,d)
# global or local variable
x = 3  # global
def f():
    x = 5  # local
    return x 

print(x)
print(f())
x = f()
print(x)

y = 6
def f():
    z = 2**y
    return z

print(y)
print(f())
# to see the builtin functions:
import builtins
dir(builtins)
# Nested functions
def circumference(x,y):
    '''find circumference of a rectengule'''
    def add(x,y):
        z = x+y
        return z
    
    return 2*add(x,y)

circum = circumference (6,7) 
# 1. Default Argument

# Here y is a default argument
def power(x,y=2):
    z = x**y
    return z

power(3) # here y is taken as 2 

power(3,4) # here we force y to be equalt to 4
# 2. Flexible Arguments
#   2.1. *args
#   It can be one argument or more 

def sum_numbers(m,*args):
    i=0
    while i<len(args):
        m += args[i]
        i+=1
    return m

sum_numbers (3,4,5)
sum_numbers (1,2,3,4,5,6,7,8,9,10)
# 2.2 **kwargs
# **kwargs is a dictionary
def dictionary (**kwargs):
    for key,value in kwargs.items():
        print(key+': ', end=' ')
        print(value)
        
dictionary (country='greece',
             capital='athens',
             population=123456789)
# Examples
cube = lambda x: x**3
cube(4)

area_of_rect = lambda a,b: a*b
area_of_rect(5,4)
# map function: takes a function and a list as an input
# applies a function to all of the items in this list

my_list = [1, 2, 3]
my_map = map(lambda x:x**2, my_list)
print(list(my_map))
name = 'mechmetkasap'
it = iter(name) # makes the name iterable

print(next(it)) # will print the first element of name
print(next(it)) # will print the second element of name
print(next(it)) # and so on
print(*it)      # will print the remaining part of the name
# ZIP
# it zips lists
list1 = [1,2,3,4]
list2 = [0,-2,4,6]
zipped = zip(list1,list2)
zip_list = list(zipped)
print(zip_list)

# UNZIP
# Note: It returns tuple insted of list
unzipped = zip(*zip_list)
unlist1, unlist2 = list(unzipped)
print(unlist1)
print(unlist2) 

print(type(unlist1))  # tuple
print(type(unlist2))  # tuple

# to make them again list we can use list() function
# LIST COMPREHENSIONS
# lets use our data now to classify skills of a player (for CR7 in our case)
# make another column for this data
# if a skill level is higher than 90: 'very high'
# if a skill level is between 90 and 80: 'high'
# if a skill level is between 80 and 60: 'medium'
# if a skill level is below 60: 'low'

data ['Skill_Levels_Ronaldo'] = ['very high' if level > 90 
     else 'high' if 90 > level > 80 else 'medium' 
     if 80 > level > 60 else 'low'
     for level in data.Christiano_Ronaldo]

data.loc[:,['Christiano_Ronaldo', 'Skill_Levels_Ronaldo']]
# it will tell us how many rows and columns we have
data.shape
# to get information about our data
data.info()
# value count
data ['Primary Skill'].value_counts() # will not count nan objects
data ['Primary Skill'].value_counts(dropna = True) # will not count nan objects
data ['Primary Skill'].value_counts(dropna = False) # WILL count nan objects
# To learn basic statistical values

data.describe()
data.boxplot(column = ['Christiano Ronaldo', 'Lionel Messi', 'Neymar'])

plt.title('Visual Exploratory Analysis of Three Player')
plt.ylabel('Level')
# 1) melt function

new_data = data.loc[4:6, :] # rows 4,5,6 of our data

melted_new_data = pd.melt(frame = new_data, id_vars = 'Primary Skill', value_vars=['Christiano Ronaldo', 'Lionel Messi'])
# 2) Pivoting data: Reverse of melting

# melted_new_data # to see our data table in order to write .pivot

melted_new_data.pivot(index = 'Primary Skill', columns = 'variable', values = 'value')

# concatenating as rows
data1 = data.loc[13:16, :] # rows 1,2,3,4
data2 = data.loc[7:10, :] # rows 7,8,9,10

# index numbers are written directly
concat_data12 = pd.concat([data1,data2], axis=0)
# index numbers are starting from 1
concat_data12_index = pd.concat([data1,data2], axis=0, ignore_index=True)
# concatenating as columns
data3 = data [data ['Christiano Ronaldo'] > 90]
#data33 = data3 ['Christiano Ronaldo]
data33 = data3.loc[:24, ['Christiano Ronaldo']]

data4 = data [data ['Lionel Messi'] > 90]
data44 = data4.loc[:26, ['Lionel Messi']]

concat_data3344 = pd.concat([data33,data44], axis=1, ignore_index=True)
# to see data types of our data
data.dtypes

# convert data type of Primary Skill from object to category
data ['Primary Skill'] = data ['Primary Skill'].astype('category')

# convert data type of Primary Skill from object to category
data ['Neymar'] = data ['Neymar'].astype('float')
# lets check new data types
data.dtypes
# now lets check do we have non null values
# normally there are no null entries, however we have put a null object into
# the 'Primary Skill' in order to work with this data 
data.info()
# to see also NaN objects, we set dropna=False
data ['Primary Skill'].value_counts(dropna=False)

# to see only non null objects we set dropna=True
data ['Primary Skill'].value_counts(dropna=True)
# to delete null objects:
data_copy = data.copy()
data_copy ['Primary Skill'].dropna(inplace=True)
# to check whether data is changed. If it is, it will return True
data_copy ['Primary Skill'].notnull().all()

# to check whether data is changed we will use assert function. If it returns
# nothing it means there is no error
assert data_copy ['Primary Skill'].notnull().all()
data_copy ['Primary Skill'].fillna('empty', inplace=True)

data_copy ['Primary Skill'].notnull().all()
# to check data types of feaures
data ['Christiano Ronaldo'].dtypes == np.int64