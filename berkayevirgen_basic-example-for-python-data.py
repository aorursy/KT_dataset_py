# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

import seaborn as sns # effective library for visualizaton

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/heart-disease-uci/heart.csv') #import data

df.info() # give me informations (please :D)!
df.corr() > 0.5 # correlationship about datas

# we cannot draw important conclusions here , lets analyse more
#correlation map

f, ax = plt.subplots(figsize=(13,11))

sns.heatmap(df.corr(), annot=True, linewidths = .5, fmt = '.1f', ax=ax)

plt.show()
print(df.head(6)) # first 6 data

print(df.tail(6)) # last 6 data

print("\n",df.columns)
df.age = df.age.astype('int')

df.describe()
# line plot

# Correlation FBS Male-Female

male = df[df.sex == 1] # male

female = df[df.sex == 0] # female

"""f, ax = plt.subplots(figsize=(15, 15))

male.fbs.plot(kind='line', color='b', label='Male', linewidth=1,grid=True, linestyle=':',)

female.fbs.plot(kind='line', color='r', label='Female', linewidth=1, grid=True, linestyle='-.')

plt.legend(loc='upper right')

plt.title('Fasting Blood Sugar Test (FBS)')

plt.xlabel('Index')

plt.ylabel('FBS Statistics')

plt.show()"""

f, ax = plt.subplots(figsize=(15, 15))

plt.plot(male.index, male.fbs, color='b', label='Male')

plt.plot(female.index, female.fbs, color='r', label='Female')

plt.title('Fasting Blood Sugar Test (FBS)')

plt.xlabel('Index')

plt.ylabel('FBS')

plt.legend(loc='upper right')

plt.show()
# Scatter Plot

# let's look correlation chol and max heart rate



df.plot(kind='scatter', x='chol', y='thalach', alpha=.5, color='r',figsize=(11, 9))

plt.xlabel('Chol')

plt.ylabel('Thalach')

plt.title('Chol-Max Heart Rate(Thalach) Scatter Plot')

plt.show()
#Histogram

# age values

df.age.plot(kind='hist', bins=100,figsize=(13,10))

plt.show()
# dictionary work

dctnry = {'Turkey' : 'Istanbul', 'Spain' : 'Granada', 'USA' : 'Las Vegas'}

print(dctnry.keys())

print(dctnry.values())

dctnry['Turkey'] = 'Izmir','Kirklareli' # changed and added one more data

print(dctnry)

dctnry['Spain'] = 'Madrid' # changed it

print(dctnry)

print('Istanbul' in dctnry) # is there any Istanbul values in dictionary? > no!

dctnry.clear() # clean it

print(dctnry)

# del dictionary > delete it

# Work some pandas library basics



series=df.age

print(type(series))

dtframe=df[['sex']]

print(type(dtframe))



#logic

print(4<2)

print(3==2)

print(5==5)

#boolean

print(True and False)

print(True or False)
# going ahead with pandas

agecholuppermean = df[np.logical_and(df.age > 54 , df.chol > 246)] #filtered two values in a dataframe

agecholuppermean.head()
# Dont forget to while and for loops :)

# while



i=0

while i != 5 :

    print(i)

    i += 1

print(i, 'is equal to 5')
# for



lis = [1,2,3,4,5]

for i in lis:

    print('i is : ',i)

print('\n')



# enumarate index and value of list

for index, value in enumerate(lis):

    print(index,':',value)

print('\n')



#using for loop with dictionaries

dictionary = {'spain':'madrid','france':'paris'}

for key, value in dictionary.items():

    print(key,':',value)

print('\n')



# we can achieve index and value with using pandas

for index, value in df[['cp']][0:5].iterrows():

    print(index,':',value)
# look some tuple

def tuple_ex():

    """ return defined t tuble"""

    t = (1,2,3)

    return t

a,b,c = tuple_ex()

print(a,b,c)
# local and global scope subject

x = 5

def f():

    x = 4

    return x

print(x) # this will return 5 > because this is global scope

print(f()) # this will return 4 > because this is local scope
# What if there is no local scope

x = 3

def f():

    y=4+x

    return y

print(f()) # there is no local scope in fuction, so it'll uses the global x
# How can we learn what is built in scope

import builtins

dir(builtins)
# We should just some look nested Func (func inside func)

def square():

    def add():

        x,y = 3,7

        z = x+y

        return z

    return add()**2

print(square())
# DEFAULT and FLEXIBLE ARGUMENTS

def f(a,b=1,c=2):

    y = a+b+c

    return y

print(f(7))

# can we change default args?

print(f(7,5,5)) # so, yes
# flexible args *args

def f(*args):

    for i in args:

        print(i)

f(55)

print('\n')

f(1,2,3,4)



# flexible args **kwargs > dictionary

def f(**kwargs):

    for key, value in kwargs.items():

        print(key, " ", value)

f(country = 'Turkey', capital='Ankara', population='Over 5 million')
# lambda func , very faster way of writing funcs

square = lambda x : x**2

print(square(4))

total = lambda x,y,z : x+y+z

print(total(5,6,8))

# lambda func , very faster way of writing funcs

square = lambda x : x**2

print(square(4))

total = lambda x,y,z : x+y+z

print(total(5,6,8))

# shoul not forget nested function (func inside func)

def square():

    def add():

        x = 5

        y = 7

        z = x + y

        return z

    return add()**2

print(square()) 
# Arguments

# default arguments

def f(a, b = 4, c = 6):

    y = a + b + c

    return y

print(f(5))

# can we change default arguments

print(f(5,8,5)) # so, yes
# flexible arguments *args

def f(*args):

    for i in args:

        print(i)

f(1)

print("\n")

f(1,2,3,4,5,6)



# flexible arguments **kwargs >> dictionary



def f(**kwargs):

    for key, value in kwargs.items():

        print(key, " ", value)

f(country = 'Turkey', capital = 'Ankara', population = 'Over 6m')
# lambda function ,really easiest way of writing function

square = lambda x: x**2     # where x is name of argument

print(square(5))

tot = lambda x,y,z: x+y+z   # where x,y,z are names of arguments

print(tot(6,7,8))
# anonym func

number_list = [1,2,4,5]

y = map(lambda x: x**2, number_list)

print(list(y))
# iterators

# iterable is an object that can return an iterator

name = 'juliano'

it = iter(name)

print(next(it)) # print next iteration

print(next(it))

print(*it) # print remaining iteration
# zip example

l1 = [2,3,4,5]

l2 = [6,7,8,9]

z = zip(l1,l2)

print(z)

z_list = list(z)

print(z_list)
# go unzip

uz=zip(*z_list)

ul1,ul2 = list(uz) # unzip returns tuple

print(ul1)

print(ul2)

print(type(ul2))
# list comprehension

# this is important topic

num1 = [1,2,3]

num2 = [i + 1 for i in num1 ]

print(num2)
# conditionals on iterable

num1 = [3,4,5]

num2 = [i**2 if i == 4 else i-3 if i < 4 else i*4 for i in num1]

print(num2)
# lets make list comp example on data

threshold = sum(df.chol)/len(df.chol)

df['chol_lv'] = ['high' if i > threshold else 'low' for i in df.chol]

print("Average of chol's : ",threshold)

df.loc[0:10,["chol_lv","chol"]]



df.head()
df.shape # it gives number of rows and columns in data

print(df.trestbps.value_counts(dropna = False)) # dont touch missing values


