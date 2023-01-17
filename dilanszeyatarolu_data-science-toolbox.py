# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

def tup_örnek():
    t=(1,2,3)
    return t
#print(tup_örnek())
a,b,c=tup_örnek()
#print(a)  --> result: "1"

#print(a,b,c)  --> result : 1 2 3

glob_var=2
def func():
    loc_var=3
    return loc_var
    
print(glob_var)  #-->global 
print(func())  #--> local
x="hello"
print(len(x))  # len(),print() : build scope

def firstfunc():
    x="world"
    def secondfunc():
        y="hello"
        return y
    return secondfunc()+" "+x
print(firstfunc())

#function inside function.
#***** DEFAULT FUNCTION*****
def default_func(x,y=10,z=20):
    z=x+y+z
    return z
print("DEFAULT FUNCTION:")
print(default_func(5))


#*****FLEXIBLE ARGUMENTS(args)*****
def flex_func(*args):
    for i in args:
        print(i)
print("FLEXIBLE ARGS:")
print(flex_func(5,10,15))


#----- (*kwargs)---------
def kwargs_fuc(**kwargs):
    for i,j in kwargs.items():
        print(i,"",j)
print("KWARGS :")
kwargs_fuc(a='aa',b='bb')
        
    
top=0
for i in range(5):
    top+=i
print(top)
    
#***lambda function *******

lambda_func=lambda x:x+x
print(lambda_func(5))
x=[1,2,3]
anonymous_func=map(lambda x:x+2,x)
print(list(anonymous_func))
a=[1,2,3]
b=[4,5,6]

c=zip(a,b)
print(type(c)) 
zip_list=list(c)
print(zip_list)


#un_zip()

un_zip=zip(*zip_list)
first_list,second_list=list(un_zip)
print(first_list)
print(second_list)

#Example 1
list1=[5,10,15]
a=[i+1for i in list1]
print(a)
#Example 2

list2=[5,10,15,20,25,30]
b=[print(str(i)+" Even number")if i%2==0 else print(str(i)+" Odd number") for i in list2]
data=pd.read_csv("../input/the-movies-dataset/movies_metadata.csv")
data.head(5)
x=data.vote_count[:10]
y=[print(i) if i>150else print("küçük") for i in x]