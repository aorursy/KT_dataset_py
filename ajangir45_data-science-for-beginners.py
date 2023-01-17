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
data=pd.read_csv('../input/pokemon.csv')
#show top 5 values of the dataset
data.head()
#all information about data
data.info()
#summarize the data
data.describe()
#show the columns of the dataset
data.columns
import matplotlib.pyplot as plt

#line plot
data.Speed.plot(kind='line',color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.Defense.plot(kind='line',color = 'g',label = 'Defense',linewidth=1,alpha = 0.5,grid = True,linestyle = '-')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('line plot')
plt.show()
#scatter plot
data.plot(kind='scatter',x='Attack',y='Defense',alpha=0.5,color='red')
plt.xlabel('Attack')
plt.ylabel('Defense')
plt.title('Scatter plot')
plt.show()
#histogram
data.hist(column='Speed',bins=50,figsize=(12,12))
dict={'India':'Delhi','France':'Paris'}
print(dict.keys())
print(dict.values())
dict['Australia']='Canberra'     #adds new entry to the dictionary
print(dict)
dict['India']='New Delhi'        #updates existing entry
print(dict)
series=data['Speed']
print(type(series))
dataframe=data[['Defense']]
print(type(dataframe))
x=data['Defense']>200
data[x]
#filtering data in pandas
data[np.logical_and(data['Defense']>200,data['Attack']<300)]
#the above code can also be written as
data[(data['Defense']>200) & (data['Attack']<300)]
i=0
while(i !=5):
    print('i is:',i)
    i+=1
print(i,'is equal to 5')
#global and local scope
x=2
def f():
    x=3
    return x
print(x)
print(f())
#nested function
def square():
    def add():
        x=2
        y=4
        z=x+y
        return z
    return add()**2
print(square())
#default and flexible arguments
def f(a,b=2,c=3):     #default
    y=a+b+c
    return y
print(f(1))
print(f(1,4,3))
#flexible arguments
def f(*args):
    for i in args:
        print(i)
f(1)
print('---------')
f(1,2,3)
def f(**kwargs):
    for key,value in kwargs.items():
        print(key,'', value)
f(country='India',capital='New Delhi')
#lambda function
square= lambda x: x**2
square(5)
#Iterators
x='Kaggle'
it=iter(x)
print(next(it))
print(*it)
#zip lists
l1=[1,2,3]
l2=[4,5,6]
l=zip(l1,l2)
print(list(l))
#list comprehension
x=[1,2,3]
y=[i+1 for i in x]
y
x=[4,10,20]
y=[i**2 if i==5 else i-5 if i<6 else i+5 for i in x]
print(y)
#performing list comprehension on our dataset
threshold=sum(data.Speed)/len(data.Speed)
data['Speed_level']=['high' if i>threshold else 'low' for i in data.Speed]
data.loc[:,['Speed_level','Speed']].head()
data=pd.read_csv('../input/pokemon.csv')
data_new=data.head()
data_new
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted=pd.melt(frame=data_new,id_vars='Name',value_vars=['Attack','Defense'])
melted
melted.pivot(index='Name',columns='variable',values='value')
data['Type 2'].value_counts(dropna=False)
#convert datatype
data['Type 1']=data['Type 1'].astype('category')
data['Speed']=data['Speed'].astype('float')
