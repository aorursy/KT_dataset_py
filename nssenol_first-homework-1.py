# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/creditcard.csv')
data.info()
data.corr()
data.head()
data.tail()
data.V1
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.columns
#MatPlotLib
data.V1.plot(kind = 'line', color = 'blue',label = 'V2',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.V2.plot(color = 'gray',label = 'V2',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
#ScatterPlot
data.plot(kind = 'scatter',x='V1',y='V2', color = 'red',alpha = 0.5,)
plt.xlabel('V1')
plt.ylabel('V2')
plt.title('Comparing V1 and V2')

#Histogram
#bins = number of bar in figure
data.V9.plot(kind ='hist',bins=30,figsize=(12,12))
plt.show()

#Burj Khalifa is shown below
data.Amount.plot(kind='hist',bins=5)
plt.clf()
dictionary = {'Turkey': 'Ankara', 'USA': 'Washington', 'Italy' : 'Rome'}
print(dictionary.keys())
print(dictionary.values())
dictionary['Turkey'] = 'Istanbul'
print(dictionary)
dictionary['france'] = 'Paris'
print(dictionary)
del dictionary['USA']
print('france' in dictionary)
dictionary.clear()
print(dictionary)
print(dictionary)
#PANDAS
data = pd.read_csv('../input/creditcard.csv')
series = data['V1']
print(type(series))
data_frame = data[['V1']]
print(type(data_frame))








print(3>2)
print(3!=2)
#Boolean Operators
print(True and False)
print(True or False)
x = data['V1']>1.5
data[x]
def tuple_ex():
    t = (1,2,3)
    return t
a,b,c = tuple_ex()
print(a,b,c)
#Scope

x=2
def f():
    x=3
    return x
print(x)
print(f())





import builtins
dir(builtins)
def square():
    def add():
        x=2
        y=5
        z=x+y
        return z
    return add()**2
print(square())
#Default Arguments
def f(a,b=1,c=2):
    y =a+b+c
    return y 
print(f(5))
print(f(5,4,3))
#Flexible Arguments
def f(*args):
    for i in args:
        print(i)
f(1)
    
def f(**kwargs):
    for key, value in kwargs.items():
        print(key, " ", value)
        f(country = 'Spain', capital = 'Madrid', Popoulation = 1234567)
        print(key, " ", value)
