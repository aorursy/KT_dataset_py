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
def tuble_x():
    
    t = (1,2,3)
    return t

a,b,c = tuble_x()
print(a,b,c)
x = 5 #global scope

def f():
    x = 4 #local scope
    return x

print(x)
print(f())

x = 14

def f():
    
    y = x**2
    return y

print(f())





# How can we learn what is built in scope
import builtins
dir(builtins)


def fonk():
    def fon():
        a = 5
        b = 4
        c = a+b
        return c
    return fon()//3

print(fonk())
        
# default argument
def f(a,b=3,c=5):
    z = a+b+c
    return z

print(f(5))
print()
print(f(3,4,5))
def h(*args):
    for i in args:
        print(i)
        
h(1,2,3)  
print()

def c(**kwargs):
    for key,value in kwargs.items():
        print(key,":",value)

c(Color = "Yellow",Fruit = "Apple",Telephone_model = "Xiaomi")        

example = lambda x: x+x*2

print(example(4))
print()

example2 = lambda x,y,z : x+y+z

print(example2(3,6,9))
liste = [3,6,9]

example3 = map(lambda x: x**2,liste)

list(example3)
name  = "Kemalettin"

f = iter(name)
print(next(f))
print()
print(*f)
#zip example

list1 = [1,2,3,4]

list2 = [5,6,7]

list3 = [7,8,9,10]

a = zip(list1,list2)
print(a)
a_list = list(a)
print(a_list)
print()
b = zip(list1,list3)

print(list(b))

x = 1

a = [i+2 for i in range(x,x+10)]

print(a)
# Conditionals on iterable

a = range(0,10)
b = [i*2 if i%2 == 1 else i+50 if  i*3 == 18 else i for i in a]
print(b)
data = pd.read_csv("../input/StudentsPerformance.csv")
threshold = int(sum(data["math score"])/len(data["math score"]))
data["Performance"] = ["good" if i > threshold else "normal" if i == threshold else "bad" for i in data["math score"]]

data.loc[:20,["Performance","math score"]]



