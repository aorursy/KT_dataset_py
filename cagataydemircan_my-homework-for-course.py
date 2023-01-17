# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/USvideos.csv")
data.columns
data.views.plot(kind = 'line', color = 'g',label = 'Views',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.likes.plot(color = 'r',label = 'Likes',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     
plt.xlabel('x axis')              
plt.ylabel('y axis')
plt.title('Views/Likes')            
plt.show()
data.views.plot(kind = "hist",bins = 50,figsize = (8,8))
plt.show()
classes = {'melee':'barbarian','ranged':'archer','magic':'templar'}
print(classes.keys())
print(classes.values())
series = data['likes']
print(type(series))
data_frame = data[['likes']]
print(type(data_frame))
for index,value in data[['likes']][0:10].iterrows():
    print(index, '-->' , value)
x = data['likes']>70000
print(x)
x = 5
def f():
    x=12
    return x
print("Global Scope:",x)
print("Local Scope:",f())
def square():
    def add():
        x = 5
        y = 5
        z = x+y
        return z
    return add()**2
print(square())
def f(a,b=2,c=3):
    return a+b+c
print(f(2))
print(f(2,4,4))
def f(*args):
    for i in args:
        print(i)
print(f(2))
print(f(2,3,4,5))
def f2(**kwargs):
    for key,value in kwargs.items():
        print(key,"-->",value )
f2(level=22,ascendant='False',ilevel=25,name="Chaux")                
sumfinder = lambda x,y,z: x+y+z
print(sumfinder(2,3,4))
num_list = [4,8,12]
y = map(lambda x:x**3,num_list)
print(y)
print(list(y))
name = "pathofexile"
it = iter(name)
print(next(it))
print(next(it))
print(*it)
list_a = [1,3,5,7]
list_b = [2,4,6,8]
z = zip(list_a,list_b)
z_list = list(z)
print(z_list)


un_zip = zip(*z_list)
newlist_a ,newlist_b = list(un_zip)
print(newlist_a)
print(newlist_b)
threshold = sum(data.likes)/len(data.likes)
print(threshold)
data["like_level"] = ["high" if i > threshold else "low" for i in data.likes]
data.loc[:100,["like_level","like"]]
