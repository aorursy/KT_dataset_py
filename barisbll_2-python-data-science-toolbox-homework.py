# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
def tuble_ex():
    """" return defined t tuble """
    t=(1,2,3)
    return t
a,b,c=tuble_ex()
print(a,b,c)
x=2
def f():
    x=3
    return(x)
print(x)
print(f())
x=5
def f():
    y=2*x
    return y
print(f())
import builtins
dir(builtins)
# nested function

def square():
    """" return square of value """
    def add():
        x=2
        y=3
        z=x+y
        return(z)
    return add()**2
print(square())
# default arguments

def f(x,y=2,z=3):
        a = x+y+z
        return(a)
print(f(5))
print(f(5,4,10))
#flexible arguments *args

def f(*args):
    for i in args:
        print(i)
f(1)
print("")
f(1,2,3,4)

# flexible arguments **kwargs that is dictionary,
def f(**kwargs):
    """" print key and value of dictionary"""
    for key,value in kwargs.items():
        print(key,"",value)
f(country="spain",value="madrid",population=123456)
        
# lambda function

square = lambda x:x**2
print(square(4)),
tot = lambda x,y,z: x+y+z
print(tot(4,5,6))
number_list=[1,2,3]
y=map(lambda x:x**2,number_list)
print(list(y))
# zip example

list1=[1,2,3,4]
list2=[5,6,7,8]
z=zip(list1,list2)
print(z)
z_list=list(z)
print(z_list)
un_zip=zip(*z_list)
un_list1,un_list2=list(un_zip)
print(un_list1)
print(un_list2)
print(type(un_list2))
# example of list comprehension

num1=[1,2,3]
num2=[i+i for i in num1]
print(num2)
num1=[5,10,15]
num2=[i**2 if i==10 else i - 5 if i<7 else i+5 for i in num1]
print(num2)
# pokemon classification

thereshold = sum(data.Speed)/len(data.Speed)
data["speed_level"]=["high" if i>thereshold else "low" for i in data.Speed]
data.loc[:10,["speed_level","Speed"]]


data = pd.read_csv('../input/Pokemon.csv')



