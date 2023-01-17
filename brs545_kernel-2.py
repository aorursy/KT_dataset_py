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
data=pd.read_csv('../input/master.csv')
data.head()
import builtins

dir (builtins)
def square():

    def add():

        x=2

        y=7

        z=9

        d= x+y+z

        return d

    return add()**2

print(square())
def f(a,b,c,d=18):

     g=a+b/c*d

     return g

print(f(3,4,5))
def f(*args):

    for i in args:

        print(i)

f(3)

print("")

f(1,2,3,4,5,6,7)

def f(**kwargs):

    for key, value in kwargs.items():

        print(key,"",value)

f(country='Sweden',capital='Stockholm',population='5344534')
square = lambda x: x**2   

print(square(6))

total = lambda x,y,z,d: x+y+z+d 

print(total(35,33,57,45))
numbers = [1,2,3]

y = map(lambda x:x**3,numbers)

print(list(y))
name = "kobe"

it = iter(name)

print(next(it))   

print(*it)
liste = [4,2,6,9]

listee = [10,1,3,7]

z = zip(liste,listee)

print(z)

z_list = list(z)

print(z_list)
un_zip = zip(*z_list)

un_liste,un_listee = list(un_zip)

print(un_liste)

print(un_listee)

print(type(un_listee))
sayı= [1,2,3]

sayıı = [i + 1 for i in sayı]

print(sayıı)
sayı = [8,14,25]

sayıı = [i**2 if i == 8 else i-7 if i < 15 else i+5 for i in sayı]

print(sayıı)
m = sum(data.population)/len(data.population)

data["population_level"] = ["high" if i > m else "low" for i in data.population]

data.loc[:10,["population_level","population"]]