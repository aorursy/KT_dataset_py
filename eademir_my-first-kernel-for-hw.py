# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statistics as s

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/creditcardfraud/creditcard.csv')
data.info()
data.corr()
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.100, fmt= '.1f',ax=ax)
plt.show()
data.head()
data.columns
data.tail(1) #there are 284806 rows in the dataset
data.V3.plot(kind = 'line', color = '#4253f4',label = 'V3',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.V4.plot(color = '#424244',label = 'V4',linewidth=1, alpha = 0.5,grid = True,linestyle = '-')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()
plt.plot(data)
plt.show()
#scatter

data.plot(kind='scatter', x='V3', y='V4',alpha = 0.5,color = 'b')
plt.xlabel('V3')            
plt.ylabel('V4')
plt.title('tenure MonthlyCharges Scatter Plot')  
plt.show()
#histogram

data.V3.plot(kind = 'hist',bins = 100,figsize = (12,12))
plt.show()
dictionary = {'names' : 'eray','nickname' : 'blue'}
print(dictionary.keys())
print(dictionary.values())
dictionary['names'] = "demir"    # update existing entry
print(dictionary)
dictionary['age'] = "25"       # Add new entry
print(dictionary)
del dictionary['names']              # remove entry with key 'name'
print(dictionary)
print('name' in dictionary)        # check include or not
dictionary.clear()                   # remove all entries in dict
print(dictionary)
del dictionary         # delete entire dictionary   
print(dictionary)      #we take error and cannot see dictionary 'cause i deleted it in the previous one line
lis = [10,12,13,15,20,29,30]

plt.plot(lis)
plt.grid(color='#545387', linestyle='-',linewidth=0.2)
plt.show()
stats.gmean(lis)      #geometric mean
s.harmonic_mean(lis)   
np.median(lis)
series = data['V4']
print(type(series))
data_frame = data[['V4']]
print(type(data_frame))
x = data['V4']>15
data[x]
data[np.logical_and(data['V4']>15, data['V3']<-20.7 )]
dictionary = {'names' : 'eray','nickname' : 'blue'}
for key,value in dictionary.items():
    print(key," : ",value)
print('')
for index,value in data[['V3']][0:1].iterrows():
    print(index," : ",value)
i = 0
while i != 5 :
    print('i is: ',i)
    i +=1 
print(i,' is equal to 5')
def tuble_ex():
    t = {1,2,3}
    return(t)
a,b,c = tuble_ex()
print(a)
print(b,c)
x = 2
def f(x):
    x = 3
    return x
print(x)    #global scope
print(f(x)) #local scope
x = 2
def f(a): #there is no x in local scope
    y = 2*x
    return y
print(f(a)) #firstly, it looks local scope and cannot find "x" then looks global scope
import builtins
dir(builtins)
def square():
    def add():
        x = 2
        y = 3
        z = 2*3
        return z
    return add()**2
print(square())
#default argument

def f(a, b=1, c=2):
    x = a+b+c
    return x
print(f(5))
print(f(5,4,3))
#flexible argument

def f(*args):
    for i in args:
        print(i)
f(1)
f(1,2,3,4)

#**kwargs dictionary

def f(**kwargs):
    for key, value in kwargs.items():
        print(key, " : ",value)
f(name = "eray",surname="demir")
    
square = lambda x: x**2
print(square(5))

total = lambda x,y,z: x+y+z
print(total(1,2,3))
n_list = (1,2,3)
asd = map(lambda x:x**2,n_list)
print(list(asd))
name="eray"
it = iter(name)
print(next(it))
print(*it)
#zip
print("zip:")
list1 = [1,2,3]
list2 = [4,5,6]
z = zip(list1,list2)
z_list = list(z)
print(z)
print(z_list)
print(type(z))
print(type(z_list))
print("")
#unzip
print("unzip:")
un_list = zip(*z_list)
un_list1, un_list2 = list(un_list)
print(un_list1)
print(un_list2)
print(type(un_list1))
print(type(list(un_list)))
num1 = [1,2,3]
num2 = [i**2 for i in num1]
print(num2)
num3 = [i**3 if i == 8 else i + 5 if i>8 else i - 1 for i in num1]
print(num3)
average = sum(data.V1)/len(data.V1)
print(average)
data["v1_average"] = ["upper" if i > average else "lower" for i in data.V1]
data.loc[:10,["V1","v1_average"]]