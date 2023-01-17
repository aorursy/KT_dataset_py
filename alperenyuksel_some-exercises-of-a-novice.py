# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

data = pd.read_csv("../input/tmdb-movie-metadata/tmdb_5000_credits.csv")

data2 = pd.read_csv("../input/tmdb-movie-metadata/tmdb_5000_movies.csv")
data2
data2.info()
data2.columns
data2.head(10)
data2.tail(10)
data2.shape
def tuble_ex():

    """return defined budgetlist"""

    budgetlist=(data2.budget.head(3))

    return budgetlist

a,b,c=tuble_ex()

print(a,b,c)

#We have taken top three budget data as a tuble using a function.
first_id=(data2.id.head(1))#ID of first movie, it's a global scope

def last():

    last_id=(data2.id.tail(1))#ID of last movie, it's a local scope

    return last_id

print(first_id)

print(last())
first_id=(data2.id.head(1))#ID of first movie, it's a global scope

def f():

    x=first_id+1

    return x

print(f())

#the situation that there is no local scope.
import builtins

dir(builtins)#if we want to learn built in scopes.
def fsum():

    """return sum of values"""

    def add():

        """add two local scope"""

        f=2

        l=3

        lsum = f+l

        return lsum

    return add()**3

print(fsum())

        
def f(a, b = 1, c = 2):

    x = a + b + c

    return x

print(f(10))

print(f(7,8,9))# we can create new values for default arguments
def f(*args):

    for i in args:

        print(i)

f(5)

print("--")

f(5,10,15,20,25)

print("----")

def f(**kwargs):

    for key, value in kwargs.items():  

        print(key, " ", value)

f(id = 28, name = 'Action')#for genres column(first one)
data2.genres[0]
cube = lambda x: x**3     

print(cube(10))

sum = lambda x,y,z: x+y+z   

print(sum(25,75,50))
n_list = [10,100,1000]

a = map(lambda x:x**3,n_list)

print(list(a))
name="Hello"
it=iter(name)

print(next(it))

print(*it) 
list1=[237000000,300000000,245000000]

list2=[19995,285,206647]

z=zip(list1,list2)

print(z)

z_list=list(z)

print(z_list)#we matched budgets and ids of the top three.
un_zip = zip(*z_list)

un_list1,un_list2 = list(un_zip)#removing the matches between budgets and ids

print(un_list1)

print(un_list2)

print(type(un_list2))
threshold =data2['budget'].sum() / len(data2)#or you can find threshold with data2['budget'].mean()

print(threshold)

data2["budget_level"] = ["high" if i > threshold else "low" for i in data2.budget]

data2.loc[:100,["budget_level","budget"]]#there is a list of top 100 movies with their budget and budget levels.