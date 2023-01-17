# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns # visualization tool

import builtins



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Iris.csv')
data.info()
data.corr()

#1'e yakın ise pozitif corr -1' e yakın ise negatif 0 ise ilişki yok.
f, ax = plt.subplots(figsize=(9,9))

sns.heatmap(data.corr(), annot= True, linewidths= .4, fmt='.1f',ax=ax)

plt.show()
data.head(15)
data.tail(15)
data.columns
#Line plot

data.SepalWidthCm.plot(kind='line',color='b',label='SepalWidthCm',linewidth=1,alpha=0.7,grid=True,linestyle=':')

data.SepalLengthCm.plot(kind='line',color='r',label='SepalLengthCm',linewidth=1,alpha=0.7,grid=True,linestyle='-.')

plt.legend()

plt.title('Line Plot')

#Scatter Plot

data.plot(kind='scatter',x='SepalLengthCm',y='PetalLengthCm',alpha=0.7,color='red',grid=True)

plt.xlabel('SepalLengthCm')

plt.ylabel('PetalLengthCm')

plt.title('SepalLengthCm-PetalLengthCm')
#Scatter Plot

data.plot(kind='scatter',x='SepalWidthCm',y='PetalWidthCm',alpha=0.7,color='blue',grid=True)

plt.xlabel('SepalWidthCm')

plt.ylabel('PetalWidthCm')

plt.title('SepalWidthCm-PetalWidthCm')
#Histogram

data.SepalLengthCm.plot(kind='hist',bins=35,figsize=(10,10))

plt.show()
#Histogram

data.PetalLengthCm.plot(kind='hist',bins=35,figsize=(10,10))

plt.show()
#Histogram

data.SepalWidthCm.plot(kind='hist',bins=35,figsize=(10,10))

plt.show()
#Histogram

data.PetalWidthCm.plot(kind='hist',bins=35,figsize=(10,10))

plt.show()
#Histogram

data.SepalLengthCm.plot(kind='hist',bins=35,figsize=(10,10))

plt.clf()
#Histogram

data.PetalLengthCm.plot(kind='hist',bins=35,figsize=(10,10))

plt.clf()
dictionary={'turkey':'ankara','england':'london'}

print(dictionary.keys())

print(dictionary.values())
dictionary['turkey']="istanbul"

print(dictionary)

dictionary['france']="paris"

print(dictionary)

del dictionary['france']

print(dictionary)

print('turkey' in dictionary)

dictionary.clear()

print(dictionary)

del dictionary
series=data['SepalWidthCm']

print(type(series))

data_frame=data[['SepalWidthCm']]

print(type(data_frame))
print(5<4)

print(5>4)

print(3 != 2)

print(3==2)

print(True or False)

print(True or True)

print(False or False)

print(True and False)

print(True and True)

print(False and False)
x= data['SepalWidthCm']>4

data[x]
data[np.logical_and(data['SepalWidthCm']>3.5,data['SepalLengthCm']>5)]  #and

#data[np.logical_or(data['SepalWidthCm']>3.5,data['SepalLengthCm']>5)]  #or

#data[data['SepalWidthCm']>3.5 & data['SepalLengthCm']>5]
i=0

while i != 5:

    print('i is : ',i)

    i+=1

print(i,'is equal to 5')
list=[1,2,3,4,5]

for i in list:

    print('i is:',i)

print(' ')



for index,value in enumerate(list):

    print(index," : ",value)

print(' ')



dictionary={'turkey':'bursa','england':'london'}

for key,value in dictionary.items():

    print(key," : ",value)

print(' ')



for index,value in data[['SepalWidthCm']][0:5].iterrows():

    print(index," : ",value)

print(' ')
#Nested Function

def square():

    def add():

        x=2

        y=3

        z=x+y

        return z

    return add()**2

print(square())
#default arguments

def f(a,b=1,c=2):

    y=a+b+c

    return y

print(f(5))

print(f(5,4,3))
#flexible arguments *args

def f(*args):

    for i in args:

        print(i)

f(1)

print("")

f(1,2,3,4)

#flexible arguments **kwargs that is dictionary

def f(**kwargs):

    for key,value in kwargs.items():

        print(key," : ",value)

f(country='spain',capital='madrid',population = 123456)

#def function

def square(x):

    return x**2

print(square(3))

def tot(x,y,z):

    return x+y+z

print(tot(1,2,3))

print(" * ")

#lambda Function

square =lambda x:x**2

print(square(3))

tot = lambda x,y,z:x+y+z

print(tot(1,2,3))
number_list = [1,2,3]

y = map(lambda x:x**2,number_list)

print(tuple(y))
#iteration example

name ="ronaldo"

it=iter(name)

print(next(it)) #print next iteration

print(*it) #print remaining iteration

# zip example

list1 = [1,2,3,4]

list2 = [5,6,7,8]

z = zip(list1,list2)

print(z)

z_list = tuple(z)

print(z_list)
un_zip=zip(*z_list)

un_list1,un_list2=tuple(un_zip)#unzip returns tuble

print(un_list1)

print(un_list2)

print(type(un_list1))
dir(builtins)
#list comprehension

num1=[1,2,3]

num2=[i+1 for i in num1]

print(num2)
num1=[5,10,15]

num2=[i**2 if i==10 else i-5 if i<10 else i+5 for i in num1]

print(num2)

threshold=sum(data.SepalWidthCm)/len(data.SepalWidthCm)

print("threshold",threshold)

data["SepalWidthCm_level"]=["high"if i>threshold else "low" for i in data.SepalWidthCm]

data.loc[:10,["SepalWidthCm_level","SepalWidthCm"]]