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
df=pd.read_csv('../input/LOL Worlds 2018 Groups stage - Player Ratings.csv')
def tuble_ex():

    t=(30,50,45,60,80,75)

    return t

a,b,c,d,e,f=tuble_ex()

print(a,b,c,d,e,f)
x=10        # Global Scope

def f():

    x=5     # Local Scope 

    y=x*2

    return y

print(x)

print(f())
import builtins     # Built Scope

dir(builtins)
def f_1():    # Nested Functions

    x=15



    def f_2():

        x=5

        y=3

        z=8

        t=x*y-z

        return t

    

    return (x**2)+f_2()

print(f_1())

def f(a=1,b=2,c=3):   # Default Arguments 

    z=a*b*c

    return z 

print(f(5,5,5))

print(f(5,5))

print(f())



print(" ")



def f_flex(*args):   # Flexible Arguments

    for i in args:

        print(i**2)

f_flex(1,2,3)



x=lambda a,b,c: a*b/c

print(x(5,25,15))

list1=df.Deaths.head(10)    

list2=df.Name.head(10)



z=zip(list2,list1)    # Zip

z_list=list(z)

print(z_list)



unzip=zip(*z_list)     # Unzip

unlist1,unlist2=list(unzip)

print(unlist1)

print(unlist2)





list3=df['Kills Total'].head(10)

z1=zip(list2,list3,list1)

z1_list=list(z1)

print(z1_list)



unzip1=zip(*z1_list)

list2,list3,list1=list(unzip1)

print(list2)

print(list3)

print(list1)
threshold=sum(df['CS Total'])/sum(df['Minutes Played'])   # List Comprehension

print(threshold)

df['CS_Average']=['high' if i>threshold else 'low' for i in df['CS Per Minute']]

df.loc[:10,['CS_Average','CS Per Minute']]