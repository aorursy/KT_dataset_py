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
data=pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
data.head()
#user defined function



def f(x):

    return x+1



print(data["V9"].tail())     #prints last 5 cells of V9 column

print(f(data["V9"].tail()))  #takes last 5 cells of V9 column into function f, then prints the results
#scope

x=data.Amount.tail()  #global



def f(x):

    x=data.Amount.head() #local

    return 10*x



print(x)     #this will use the global x variable

print(f(1))  #function f will use the local x variable within its scope
def f():

    return a**2      

print(f())         #an error will rise since x is not defined in global, local or builtin scope
#list of builtin functions

import builtins

dir(builtins)
#using builtin scope

def f():

    if False:

        return 10

    else:

        return 20

print(f())
x=data.loc[0:2,"Time"]

type(x)
#nested function

def f():

    x=(data.loc[:,"Time":"V9"]).corr()    #correlation table of first 10 columns of credit card data

    def g():

        a=10

        return a*x               # g() returns 10*x

    return g()*10                # f() returns 100*x



print(x.sum())

print(f().sum())
#nestedfunction

import math

def f(x):

    def square(x):

        return x**2

    return math.sqrt(square(x))

print(f(15))
#default and flexible arguments

print(data.loc[0,"Amount"])

print(data.loc[1,"Amount"])



def f(a,b=2,c=3):

    return a*b*c

print(f(1))             #returns 1*2*3

print(f(1,data.loc[0,"Amount"]))       #returns 1*149.62*3

print(f(1,data.loc[0,"Amount"],data.loc[1,"Amount"]))   #returns 1*149.62*2.69

print(f(5,10,15))        #returns 5*10*15
#flexible arguments

def f(*args):

    for i in args:

        print(i)

print(f(data.loc[0,:]))      #prints index 0 of credit card data
#flexible arguments

def f(**kwargs):

    for key,value in kwargs.items():

        print(key,":",value)

f(spain="madrid",brittain="londra",usa="washington")
#lambda function

multiplied_by_2=lambda x:2*x

print(data.loc[0:10,"Time"])

print(multiplied_by_2(data.loc[0:10,"Time"]))
#anonymous function: like lambda function but it can take more than one argument

#map function: applies a function to all the items in a list. 

#              returns a map object, this object can be converted into a list.



list1=list(data.loc[:,"Time"])    #Time column of the original dataframe is converted into a list.

a=map(lambda x:2*x,list1)         



print(a)                    # prints the map object itself

print(sum(list1))           # prints sum of values of 'list1'

b=list(a)                   # map object 'a' is converted into a list, named 'b'

print(sum(b))               # prints sum of values of 'b'

print(sum(b)/sum(list1))    # this is expected to print '2' since our map object was created by function "lambda x:2*x"
#iterators

data1=data.loc[0:2,"Time":"V2"]

dict1=dict(data1)



for key,value in dict1.items():

   print("The values of",key,"are:")

   print(value)
#zip function

list1=list(data.loc[0:6,"Time"])

list2=list(data.loc[0:6,"Amount"])

a=zip(list1,list2)

print(a)

b=list(a)

print(list1)

print(list2)

print(b)
#unzip function

a=zip(list1,list2)

unzip=zip(*a)

unlist1,unlist2=list(unzip)    #returns tuple

print(unlist1)

print(unlist2)

print(type(unlist1))



unlist1=list(unlist1)      #converts unlist1 into a list

print(type(unlist1))
#list comprehension

list1=list(data.loc[0:9,"Amount"])

print(type(list1))

list1.append(87.08)



average_amount=sum(list1)/len(list1)

print("average amount:",average_amount)

print("list1:")



print(list1)

list2=["high amount" if i>average_amount else "low amount" if i<average_amount else "average amount" for i in list1]

a=zip(list1,list2)

print("")

print(list(a))
#list comprehension, adding a new column to dataframe

average_amount=data.Amount.sum()/len(data.Amount)

print(average_amount)

data["Amount_type"]=["high" if i>average_amount else "low" if i<average_amount else "average" for i in data.Amount]

print(data.head())