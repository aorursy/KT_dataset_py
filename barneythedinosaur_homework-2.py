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

data = pd.read_csv('../input/kc_house_data.csv')


data.info()
data.columns
#User Defined Function
def tublee():
    x=(8,9,5)
    return x
a,b,c =tublee()
print(a,b,c)
#Scope
x=5
def f():
    x=4
    return x
print(x)
print(f())
#Nested Functions
def cube():
    def summ():
        x=2
        y=5
        z=x*y
        return z
    return summ()**3
print(cube())
    
#Default Argument
def test(x,y=12,z=8,e=6):
    a=(y-z+e)/x
    return a 
print(test(5))
print(test(6,5,4,3))
#Flexible Argument
def f(*args):
    for i in args:
        print(i)
f(1,3,5)
#print("")
#f(1,2,3,4)
def f(**kwargs):

    for key, value in kwargs.items():            
        print(key, " ", value)
f(name= 'lisa', age = 17, surname = "simpson")
#Lambda Function
cube= lambda x:x**3
print(cube(3))
total=lambda x,y,z:x+y+z
print(total(5,6,9))
#Anonymous Function
liste=[5,6,7,8]
y=map(lambda a:a**3,liste)
print(list(y))
#Iterators
name="lisa"
it=iter(name)
print(next(it))
print(*it)

#Zip 
list1=[2,6,8,9,16]
list2=[5,3,1,4,12]
z=zip(list1,list2)
#print(z)
z_list=list(z)
print(z_list)

#Unzip
un_zip=zip(*z_list)
un_list1,un_list2=list(un_zip)
print(un_list1)
print(un_list2)
print(type(un_list1))
#List Comprehension
num1=[8,9,6,5]
num2=[i*5 for i in num1]
print(num2)
thresold=sum(data.bedrooms)/len(data.bedrooms)
data["condition"]=["enough"if i>thresold else "not adequate" for i in data.bedrooms]
data.loc[:20,["condition","bedrooms"]] # we will learn loc more detailed later
