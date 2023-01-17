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
def udf():
    """User Defined Function"""
    print("This is a user defined Function")
#Scope

x = 10 #Global Scope Variable

def function():
    y = 2#Local Scope Variable

import builtins #built-in scope
dir(builtins)
def nst():
    """Nested Function"""
    x = 10
    
    def nstd(y) :
        return y**2
    
    return nstd(x)

print(nst())
def f(a,pi=3.14):
    """Default Function"""
    return a*a*pi

def g(a,b,*args):
    """Flexible Function"""
    print(*args)
    
g(1,2,3,4,5,"asdsad")
lmbd = lambda x,y : y + x*x#lambda function

lst = [1,3,5,7]

a = map(lambda b : b*3,lst)

print(list(a))
name = "abcdef"

it = iter(name)

print(next(it))

next(it)

print(*it)
fib1 = [1,2,5,13]
fib2 = [1,3,8,21]

fib = zip(fib1,fib2)#zip

fibl = list(fib)

print(fibl)
unfib = zip(*fibl)#unzip 

unfib1,unfib2 = list(unfib)

print(unfib1)

print(unfib2)

print(type(unfib2))
#list comprehensive
number = [1,2,3,4,5,6,7,8,9]

even = [True if a%2==0 else False for a in number]

even