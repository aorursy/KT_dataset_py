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
data = pd.read_csv("../input/globalterrorismdb_0718dist.csv",encoding='ISO-8859-1')
#USER DEFINED FUNCTION
def function():
    a = (23,54,76)
    return a
x,y,z = function()
print(x,y,z)
#SCOPE
a = 23
def f():
    a = 3*3/5
    return a
print(a)   #a = 23 global scope
print(f()) #a = 1.8 local scope
#WITHOUT LOCAL SCOPE
s = 13
def f():
    d = s * s
    return d
print(f())
#BUILT IN SCOPE
import builtins
dir(builtins)
#NESTED FUNCTION
def divide():
    #return divided value
    def multiplication():
        x = 4
        y = 9
        return x * y
    return multiplication()/3
print(divide())
#DEFAULT ARGUMENTS
def f(x, y = 3, z = 4):
    a = x + y * z
    return a
print(f(6))
print(f(23,1,3))
#FLEXİBLE ARGUMETNS
def f(*args):
    for i in args:
        print(i)
f(1)
print("\n")
f(3,4,5,6)

def f(**kwargs):
    #print teams and countries of teams
    for key, value in kwargs.items():
        print(key," ",value)
f(teams = ["Besiktas JK","Real Madrid"], countries = ["Turkey","Spain"])
#LAMBDA FUNCTION
add = lambda x: x + 22
print(add(23))
multiplication = lambda a,b,c: a * b * c
print(multiplication(3,4,5))
#ANONYMOUS FUNCTION
numberList = [12,14,15]
y = map(lambda a: a**2,numberList)
print(list(y))
#ITERATORS
country = "new zealand"
i = iter(country)
print(next(i))
print(*i)
list1 = [12,13,14,15]
list2 = [21,22,23,24]
z = zip(list1,list2)
print(z)
zList = list(z)
print(zList)
unZip = zip(*zList)
unZip1,unZip2 = list(unZip)
print(unZip1)
print(unZip2)
print(type(unZip1))
print(type(list(unZip1)))
#LIST COMPREHENSION
num1 = [2,33,45]
num2 = [i * 2 for i in num1]
print(num2)
num1 = [4,17,10]
num2 = [i**2 if i > 16 else i-5 if i < 7 else i+5 for i in num1]
print(num2)
threshold = sum(data.country)/len(data.country)
data["region"] = ["high" if i > threshold else "low" for i in data.country]
data.loc[:10,["region","country"]] 