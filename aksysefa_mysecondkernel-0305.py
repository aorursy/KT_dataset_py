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
data1=pd.read_csv('../input/tmdb_5000_credits.csv')
data2=pd.read_csv('../input/tmdb_5000_movies.csv')
data1.info()
data2.info()
data1.corr()
data2.corr()
f,ax=plt.subplots(figsize=(17,17))
sns.heatmap(data2.corr(),annot=True,linewidths=5,fmt='.1f',ax=ax)
plt.show()
data2.head()
data2.columns
# line plot

data2.revenue.plot(kind='line',color='r',label='revenue', linewidth=1,grid=True,alpha=0.5,linestyle=':')
data2.budget.plot(kind='line',color='g',label='budget', linewidth=1,grid=True,alpha=0.5,linestyle='-.')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('line plot')
plt.show()
# scatter plot
data2.plot(kind='scatter',x='budget',y='revenue',color='r')
plt.xlabel('budget')
plt.ylabel('revenue')
plt.title('budget and revenue scatter plot')


data2.revenue.plot(kind='hist',bins=50,figsize=(12,12))
plt.show()
data2.revenue.plot(kind = 'hist',bins = 50)
plt.clf()
series=data2['budget']
print(type(series))
data_frame=data2[['budget']]
print(type(data_frame))
x=data2['budget']>270000000
data2[x]
data2[np.logical_and(data2['revenue']>933959197,data2['budget']>270000000)]
data2[(data2['revenue']>933959197)&(data2['budget']>270000000)]
lis=[1,2,3,4,5]
for i in lis:
    print('i is:',i)
    print(':')
    
    for index,value in enumerate(lis):
        print(index,':',value)
        
        dictionary = {'spain':'madrid','italy':'roma'}
for key,value in dictionary.items():
    print(key," : ",value)
print('')
        
      

def tuble_ex():
    t=(1,3,4)
    return t
a,b,c=tuble_ex()
print(a,b,c)
    
import builtins
dir(builtins)
def f(*args):
    for i in args:
        print(i)
f(1)
print("")
f(1,2,3,4)

def f(**kwargs):
    for key,value in kwargs.items():
        print(key,value)
        f(country = 'spain', capital = 'madrid', population = 123456)
    
square=lambda x:x**2
print(square(3))
total=lambda x,y,z:x+y+z
total(1,2,3)
number_list = [2,5,7]
y = map(lambda x:x**2,number_list)
print(list(y))
name="mahmut"
it=iter(name)
print(next(it))
print(*it)
num1 = [1,2,3]
num2 = [i + 1 for i in num1 ]
print(num2)
num1=[3,5,2]
num2=[i**2 if i==3 else i-1 if i>4 else i+2 for i in num1  ]
print(num2)
threshold=sum(data2.revenue)/len(data2.revenue)
data2["revenue level"]=["high" if i>threshold else "low" for i in data2.revenue]
data2.loc[:10,["revenue_level","revenue"]]