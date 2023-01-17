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
data = pd.read_csv('../input/StudentsPerformance.csv')
data.head()
def tub():

    num = (10,20,30)    #local scope

    return num

a,b,c = tub() 

import builtins   

dir(builtins)   #built in scope
#nested function and lambda function

z=2 #global scope

def square():

    val = lambda a=5,b=6,c=7:a*b*c #lambda function

    return val()**2

print(square())
def f(a,b,c=30):     #c is a default function

    z = a*b-c

    return z

print(f(2,4))



def fas(*args):

    for i in args:

        z = i+1

    return z

fas(3,2,1)

def ase(**kwargs):

    for key,value in kwargs.items():

        print(key,':',value)

ase(country = 'İtalian',city = 'Napoli')
values = [3,6,7]

p = map(lambda x:x**2,values)

print(p)

print(list(p))
country = 'İtalian'

it_country = iter(country)

print(next(it_country))

print(*it_country)

#zip method

liste = [2,3,2,1]

lis = [6,7,9,3,7]

a = zip(liste,lis)

cl =list(a)                   

print(cl)
#unzip

un_zip_cl = zip(*cl)

liste,lis = list(un_zip_cl)

print(liste)

print(lis)   #7 is dead
s = [10,11,12]

s1 = [i+1 for i in s]

print(s1)
ı = [10,2,20]

c = [i*5 if i <10 else i/2 if i>10 else i  for i in ı]

print(c)

t = sum(data['math score'])/len(data['math score'])

data['Math level'] = ['good' if i>t else 'bad'  for i in data['math score']]

data.loc[:10,['math score','Math level']]

                                     