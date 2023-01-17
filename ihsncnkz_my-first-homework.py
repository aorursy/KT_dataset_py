# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/data.csv')
data.info()
data.corr()
f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)

plt.show()
data.head(10)
data.columns
data.Age.plot(kind='line',color='b',label='Age',linewidth=1,alpha=0.5,grid=True,linestyle=':')

data.Potential.plot(color='r',label='Potential',linewidth=1,alpha=0.5,grid=True,linestyle="-.")

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
data.plot(kind='scatter',x='Age',y='Potential',alpha=0.5,color='blue')

plt.xlabel('Age')

plt.ylabel("Potential")

plt.title('Age Potential Scatter Plot')

plt.show()
data.Age.plot(kind='hist',bins=50,figsize=(12,12))

plt.show()
data.Age.plot(kind='hist',bins=50)

plt.clf()
dictionary = {'Name' : 'Messi','Position' : 'RF'}

print(dictionary.keys())

print(dictionary.values())
dictionary['Name'] = "Barcelona"    

print(dictionary)

dictionary['country'] = "Argentina"       

print(dictionary)

del dictionary['Name']             

print(dictionary)

print('country' in dictionary)       

dictionary.clear()                  

print(dictionary)
print(dictionary) 
data=pd.read_csv('../input/data.csv')
series = data['Potential']

print(type(series))

data_frame=data[['Potential']]

print(type(data_frame))
print(3>2)

print(3!=2)



print(True and False)

print(True or False)
x =data['Potential']>90



data[x]
data[np.logical_and(data['Potential']>90,data['Age']<20)]
data[(data['Potential']>90) & (data['Age']<20)]
i = 0

while i != 5 :

    print('i is: ',i)

    i +=1 

print(i,' is equal to 5')
lis = [1,2,3,4,5]

for i in lis:

    print('i is: ',i)

print('')



for index, value in enumerate(lis):

    print(index," : ",value)

print('')



dictionary = {'spain':'madrid','france':'paris'}

for key,value in dictionary.items():

    print(key," : ",value)

print('')



for index,value in data[['Potential']][0:1].iterrows():

    print(index," : ",value)
def tuble_ex():

    t=(1,2,3)

    return t

a,b,c=tuble_ex()

print(a,b,c)
x=2

def f():

    x=3

    return x

print(x)

print(f())
x=5

def f():

    y=2*x

    return y

print(f())
import builtins

dir(builtins)
def square():

    def add():

        x=2

        y=3

        z=x+y

        return z

    return add()**2

print(square())
def f(a,b=1,c=2):

    y=a+b+c

    return y

print(f(5))

print(f(5,4,3))
def f(*args):

    for i in args:

        print(i)

f(1)

print("")

f(1,2,3,4)



def f(**kwargs):

    for key, value in kwargs.items():

        print (key," ",value)

f(country='spain',capital='madrid',population=123456)
square=lambda x:x**2

print(square(4))

tot=lambda x,y,z: x+y+z

print(tot(1,2,3))
number_list = [1,2,3]

y = map(lambda x:x**2,number_list)

print(list(y))
name="ronaldo"

it = iter(name)

print(next(it))

print(*it)
list1=[1,2,3,4]

list2=[5,6,7,8]

z=zip(list1,list2)

print(z)

z_list=list(z)

print(z_list)
un_zip = zip(*z_list)

un_list1,un_list2 = list(un_zip)

print(un_list1)

print(un_list2)

print(type(un_list2))
num1=[1,2,3]

num2=[i+1 for i in num1 ]

print(num2)
num1=[5,10,15]

num2=[i**2 if i==10 else i-5 if i<7 else i+5 for i in num1]

print(num2)
Agemean=sum(data.Age)/len(data.Age)

data["Age_level"]=["Elderly" if i>Agemean else "Young" for i in data.Age]

print(Agemean)

data.loc[:10,["Age_level","Age"]]
data=pd.read_csv('../input/data.csv')

data.head()
data.tail()
data.columns
data.shape
data.info()
print(data['Name'].value_counts(dropna=False))
data.describe()
data.boxplot(column='Age',by='Potential',figsize=(12,12))
data_new = data.head()

data_new
melted=pd.melt(frame=data_new,id_vars='Name',value_vars=["Age","Potential"])

melted
melted.pivot(index = "Name",columns="variable",values="value")
data1=data.head()

data2=data.tail()

conc_data_row=pd.concat([data1,data2],axis=0,ignore_index=True)

conc_data_row
data1=data["Age"].head()

data2=data["Potential"].head()

conc_data_col=pd.concat([data1,data2],axis=1)

conc_data_col
data.dtypes
data["Name"]=data["Name"].astype("category")

data["Age"]=data["Age"].astype("float")
data.dtypes
data.info()
data["Age"].value_counts(dropna=False)
data1=data

data1["Age"].dropna(inplace=True)
assert 1==1
assert data["Age"].notnull().all()
data["Age"].fillna("empty",inplace=True)
assert data["Age"].notnull().all()