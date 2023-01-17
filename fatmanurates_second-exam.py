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
def tublee():
    """return tub"""
    tub = ('dog','cat','horse')
    tub2 = (4,5,6)
    return tub,tub2
print(tublee())
r = 5       #global scope
def calculating_circle_area():
    r = 7           #local scope
    pi = 3.14       
    area = pi*r*r
    return area
print(calculating_circle_area())
r = 5
def calculating_circle_area():      
    pi = 3.14       
    area = pi*r*r
    return area
print(calculating_circle_area())
import builtins
dir(builtins)
def circle_area():
    r = 5
    def result():
        pi = 3.14
        a = pi*r*r
        return a
    return result()
print(circle_area())
def triangel_area(a ,h = 5):
    calculate = (a*h)/2
    return calculate
print(triangel_area(2))
# what if we want to change default arguments
print(triangel_area(7,5))
def f(*args):
    for i in args:
        i = i*i*i
        print(i)
f(5)
f(6,2,1)
def f(**kwargs):
    for key, value in kwargs.items():
        print(key," ",value)
f(color = 'black', mean = 'siyah', feature = 'color of darkness')
circle_area = lambda pi,r : pi*r*r
print(circle_area(3.14,4))
square = lambda x : x**2
print(square(7))
date_of_birth = [1994,1995,1996,1997]
a = map(lambda x:x+7, date_of_birth)
print(list(a))
country = "TÜRKİYE"
my_country = iter(country)
print(next(my_country))
print(next(my_country))
print(next(my_country))
print(next(my_country))
print(next(my_country))
print(next(my_country))
print(next(my_country))
country = "TÜRKİYE"
my_country = iter(country)
print(*my_country)
list1 = ['black','purple','pink']
list2 = ['siyah','mor','pembe']
z = zip(list1,list2)
print(z)
z_list=list(z)
print(z_list)
un_zip = zip(*z_list)
un_zip1,un_zip2 = list(un_zip)
print(un_zip1)
print(un_zip2)
print(type(un_zip2))
list1 = [1,2,3,4,5]
list2 = [i*2 for i in list1]
print(list1,list2)
list1 = [1,2,3,4,5]
list2 = [i*2 if i<3 else i-3 if i==3 else i/2 for i in list1]
print(list2)