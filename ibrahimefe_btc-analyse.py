# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
import datetime
import matplotlib
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/bitflyerJPY_1-min_data_2017-07-04_to_2018-06-27.csv")
data.Timestamp = pd.to_datetime(data.Timestamp, unit='s') #timestamp to date converting
data.info()
data1.describe()
data.head(10)
data.corr()# show us to which column relation with other 1 is most
f,ax=plt.subplots(figsize=(15,15))#managing the table size
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()
data.columns
#line plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.Low.plot(kind='line', color='g', label='Low',linewidth=2,alpha = 0.5,grid = True,linestyle = ':')
data.High.plot(color = 'r',label = 'High',linewidth=1, alpha = 0.5,grid = True,linestyle = ':')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
#plt.scatter(data.Low,data.High)
#data = data.rename(columns={'Volume_(Currency)': 'Volume_Currency'})
data.columns = data.columns.str.replace('\(','')
data.columns = data.columns.str.replace('\)','')
data.columns
# Scatter Plot 
# x = Volume_(BTC), y = High
data.plot(kind='scatter', x='Volume_Currency', y='Volume_BTC',alpha = 0.5,color = 'b')

plt.xlabel('Volume_Currency')              # label = name of label
plt.ylabel('Volume_BTC')
plt.title('Volume_Currency - Volume_BTC Scatter Plot')   
# Histogram
# bins = number of bar in figure
data.Volume_BTC.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
# 1 - Filtering Pandas data frame
x = data['Volume_BTC']>400.5   
data[x]
#we have two way to filtering 
#data[(data['Volume_BTC']>400.5) & (data['Weighted_Price']>1044666.710220)] 
data[np.logical_and(data['Volume_BTC']>400.5, data['Weighted_Price']>1044666.710 )]
for i in data:
    print('i is: ',data[i])
print('')
# For pandas we can achieve index and value 
#generate first ten rows ([0:10])
for index,value in data[['Low']][0:10].iterrows():
    print(index," : ",value)
def tuble_ex():
    """ return defined t tuble"""
    t = (1,2,3)#i just learn accidentally when you write this: (1,2,a) and then print = 1 2 1
    # but you want to print a and then  t = (1,2,'a')
    return t
a,b,c = tuble_ex()# and then print = 1 2 1
print(a,b,c)
x=2
def f():
    x=3
    return x
print(x)
print (f()) #guess what?
import builtins # How can we learn what is built in scope
dir(builtins)
def f(a,b=1,c=2):
    y=a+b+c
    return y
print (f(5)) #this case b and c is default a is flexible and it is 5.
print(f(5,4,3))
def f (*args):
    for i in args:
        print (i)
f(1)
f(1,2,3,4)
# flexible arguments **kwargs that is dictionary
def f(**kwargs):
    for key,value in kwargs.items():
        print (key," ",value)
f(country = 'spain', capital = 'madrid', population = 123456)
square=lambda x:x**2
print(square(4))

tp=lambda a,b,c:a+b+c
print (tp(1,2,3))
num_lst=[1,2,3]
y=map(lambda x:x**2,num_lst)
print (list(y))
name="istanbul"
it=iter(name)
print (next(it))#print first char
print(*it)#print remaining
print (it)
# zip example
list1 = [1,2,3,4]
list2 = [5,6,7,8]
z=zip(list1,list2)
print(z)
z_list = list(z)
print(z_list)
un_zip = zip(*z_list)
un_list1,un_list2 = list(un_zip) # unzip returns tuble
print(un_list1)
print(un_list2)
print(type(un_list2))
num1 = [1,2,3]
num2=[i+1 for i in num1]
print (num2)
# Conditionals on iterable
nm1 = [5,10,15]
nm2 =[i**2 if i==10 else i-5 if i<7 else i+5 for i in nm1]
print (nm2)
