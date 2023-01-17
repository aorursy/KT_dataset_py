# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# user defined function

def calculate(x,y):

    a = x+y**y

    return a 

print(calculate(3,4))
# global: defined main body in script

t = 8

p = 6

def ozy(k):

    p = 2

    t = 4

    r = (k+t)**p

    return r

print(ozy(3))
# usable ones are located in function

t = 82347     #not important for function

p = 6234708    #not important for function

def ozy(k):

    p = 2

    t = 4

    r = (k+t)**p

    return r

print(ozy(3))
#local: defined in a function

a = 2

def f():

    q = a/10

    return q

print(f())
# nested function

y = 5

x = 8

def f():

    def t():

        x = y**2

        return x

    return t()/2

print(f())

        
# default arguments

def f(a,b,c=2):

    result = (a**b)/c**2

    return result

print (f(2,3))

# what if we want to change default arguments

print (f(1,1,1))
# flexible arguments *args

def f(*args):

    for i in args:

        print(i**2)

f(1,2,3)



def q(*args):

    for i in args:

        a = 2

        print(i**a)

q(1,2,3)

# flexible arguments **kwargs that is dictionary

def t(**kwargs):

    for key,value in kwargs.items():

        print(key," ",value)

t(country = 'Turkey', capital = 'Ankara', population = 5639076)
# lambda function

bomb = lambda x: x**x

print(bomb(5))

hey = lambda a,b,c: a*b*c

print(hey(1,2,3))
#ANONYMOUS FUNCTİON

list_of_numbers = [1,2,3,4,5,6,7,8,9]

x = map(lambda a: a**2,list_of_numbers)

print(list(x))
# zip example

list1 = [1,2,3]

list2 = [1,4,9]

a = zip(list1,list2)

print(a)

a_list = list(a)

print(a_list)
un_zip = zip(*a_list)

un_list1,un_list2 = list(un_zip) 

print(un_list1)

print(un_list2)

print(type(un_list2))
# list comprehension

a = [1,2,3]

b = [i + i**2 for i in a]

print(b)
# Conditionals on iterable

x = [100,1000,10000]

y = [i*0.5 if i<101 else i*5 if i<1001 else i*50 for i in x]

print(y)