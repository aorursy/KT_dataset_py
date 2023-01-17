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
data=pd.read_csv('../input/Iris.csv')
def tuble_ex():

    t=(1,2,3)

    return t

a,b,c=tuble_ex()

print(a,b,c)
#Guess Print What

x=2

def f():

    x=3

    return x

print(x)

print(f())
#What if there is no local scope

x=5

def f():

    y=2*x

    return y

print(f())

import builtins

dir(builtins)
#Nested Functions

def square():

    def add():

        x=2

        y=3

        z=x+y

        return z

    return add()**2

print(square())

#Default Arguments

def f(a,b=1,c=2):

    y=a+b+c

    return y

print(f(5))

print(f(5,4,3))

#Flexible Argument *Args

def f(*args):

    for i in args:

        print(i)

f(1)

print("")

f(1,2,3,4)
def f(**kwargs):

    for key,value in kwargs.items():

        print(key,"",value)

f(country='Spain',Capital='Madrid',popuation=123456)
#Square Function

square=lambda x: x**2

print(square(4))



tot=lambda x,y,z: x+y+z

print(tot(1,2,3))
#Anonymus Function

number_list=[1,2,3]

y=map(lambda x:x**2,number_list)

print(list(y))
#Iterators

name="ronaldo"

it=iter(name)

print(next(it))

print(*it)
#Zip Example

list1=[1,2,3,4]

list2=[5,6,7,8]

z=zip(list1,list2)

print(z)

z_list=list(z)

print(z_list)
unzip=zip(*z_list)

un_list1,un_list2=list(unzip) #Unzip Returns Tuble

print(un_list1)

print(un_list2)

print(type(un_list2))

print(type(list(un_list1)))
#List Comprehension

num1=[1,2,3]

num2=[i+1 for i in num1]

print(num2)
#Conditionals on Iterable

num1=[5,10,15]

num2=[i**2 if i==10 else i-5 if i<7 else i+5 for i in num1 ]

print(num2)
