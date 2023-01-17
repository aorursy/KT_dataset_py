# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def tuble_ex():

    t = (1,2,'b','c')

    return t

a,b,c,d = tuble_ex()

print(a,b,c,d)
import builtins

dir(builtins)
#nested function

def square():

    """ return square of value """

    def add():

        """ add two local variable """

        x = int(input("Enter first number:"))

        y = int(input("Enter second number:"))

        sum = x+y

        print("Sum of",x,"and", y," is equal  to",sum)

        return sum

    return add()**2



print("Square of sum is equal to",square()) 
# default arguments

def f(a, b = 1, c = 2):

    y = a + b + c

    return y

x = int(input("Enter a number:"))

print(f(x))

# what if we want to change default arguments

print(f(5,4,3))
# flexible arguments *args

def f(*args):

    for i in args:

        print(i)

f(1)

print("")

f(1,2,3,4)

# flexible arguments **kwargs that is dictionary



def f(**kwargs):

    """ print key and value of dictionary"""

    for key, value in kwargs.items():               # If you do not understand this part turn for loop part and look at dictionary in for loop

        print(key, " ", value);



f(Name1 = 'ali',Name2 = "veli",Name3 = "kenan")

# lambda function

square = lambda x:x ** 2    # where x is name of argument

print(square(4))

tot = lambda x,y,z:x+y+z  # where x,y,z are names of arguments

print(tot(1,2,3))
list1 = []

for x in range(10):

    list1.append(x)

print(list1)



y = map(lambda x:x**3,list1)

print(list(y))
# iteration example

name = input("Enter your string:")

it = iter(name)

print(next(it))    # print next iteration

print(*it)         # print remaining iteration
# zip example

list1 = []

list2 = []

for x in range(10):

    list1.append(x**2)

    list2.append(x**3)

z = zip(list1,list2)

print(z)

z_list = list(z)

print(z_list)
un_zip = zip(*z_list)

un_list1,un_list2 = list(un_zip) # unzip returns tuble

print(un_list1)

print(un_list2)

print(type(un_list2))
# Example of list comprehension

num1 = []

for x in range(10):

    num1.append(x**2)



print(num1)

num2 = [i * 3 for i in num1 ]

print(num2)
# Conditionals on iterable

num1 = []

for x in range(10):

    num1.append(x*5)



num2 = [i**2 if i%5 == 0 else i-5 if i < 7 else i+5 for i in num1]

print(num2)
# lets return pokemon csv and make one more list comprehension example

# lets classify pokemons whether they have high or low speed. Our threshold is average speed.

data = pd.read_csv("../input/creditcard.csv")

data.info()

data.describe()

threshold = sum(data.Amount)/len(data.Amount)

data["Amount_level"] = ["high" if i > threshold else "low" for i in data.Amount]

data.loc[:10,["Amount_level","Amount"]] # we will learn loc more detailed later