# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #basic data visualization

import seaborn as sns #data visualization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/gpu-runtime/sgemm_product.csv')
data.head()
data.info()
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt='.1f',ax=ax)

plt.show()
data.columns #mwg,vwn
#line

data.MWG.plot(kind ='line',alpha=1,grid=True,color='r',label='MWG',linestyle=':')

data.VWN.plot(color='g',label='VWN',linestyle='-.',alpha=0.7,grid=True)

plt.legend(loc ='right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
#Scatter Plot

data.plot(kind='scatter',x='MWG',y='Run2 (ms)',alpha=0.7,color='g')

plt.title('ow man')

plt.show()
#histogram

data.VWN.plot(kind='hist',bins=50,figsize=(12,12))

plt.show()
# clf()

data.VWN.plot(kind = 'hist',bins =50)

plt.clf() #dlt plot

plt.show()
#dictionary

dic={'a':'ali','b':'barış'}

print(dic.keys())

print(dic.values())
dic
dic['c']='ceylan' #new dictinory
dic
del dic['a'] #remove dic element
dic
dic.clear()
dic
mwg=data['MWG']
mwg
mvg = data[['MWG']]
mvg
x = data['MWG']>200
data[x]
data.MWG
data.KWG
data[(data['MWG']>66) & (data['KWG']>20)] #(data['MWG']>66) and (data['KWG']>20)
i=9

while i != 0:

    print('i is :',i)

    i-=1
lis=[1,8,9,2,6]

y=0

for i in lis:

    print(y,'.indis =',i)

    y+=1
dec={'a':'ali','b':'bayram','c':'celil','d':'deniz'}
dec
for key,value in dec.items():

    print(key," : ",value)
def example():

    t =(1,2,3)

    return t



a,b,c = example()

print(a,b,c)

    
#phyton static execute

def a():

    print(x)

    

def z():

    x=5

    a()

x=4

z()



import pandas

dir(pandas)
def f(*args):

    for i in args:

        print(i)

f(1,2)

f(5,1,9)
add =lambda x,y,z : x+y+z

add(1,2,3)
number_list = [1]
name = "ronaldo"

it = iter(name)

print(next(it))

print(*it)
list1 =[1,2,3,4]

list2 =[5,6,7,8]

z = zip(list1,list2)

print(z)
z_list = list(z)

print(z_list)
un_zip =zip(*z_list)

un_list1,un_list2 = list(un_zip)

print(un_list1)

print(un_list2)

print(type(un_list2))
dict={'spain':'madrid','france':'paris'}

for key,value in dict.items():

    print(key,' : ',value)
#anonymous Function =Like lamda functin but can take more than one arguments.

number_list =[1,2,3]

y = map(lambda x:x**2,number_list)

print(list(y))
#Example of list comprehension

num1 = [1,2,3]

num2 =[i+1 for i in num1]

print(num2)
num1 = [5,10,15]

num2 =[1 if i<7 else i+5 for i in num1 ]

print(num2)
num1 = [10,15,6]

num2 =[i**2 if i == 10 else i-5 if i<7 else i+5 for i in num1 ]

print(num2)

#one = left if(i == 10) =true(two),false (three)

#two = i**2

#three = right if(i<7) =true(four),false(five)

#four = i-5 (left else)

#five =i+5(right else)
data.columns
data['MWG']

threshold = sum(data.MWG)/len(data.MWG)

data['MWG_Level']=['high' if i > threshold else 'low' for i in data.MWG]

data.loc[:10,["MWG",'MWG_Level']]
threshold
data.shape
data.info()
data.columns
data.info()
print(data['MWG_Level'].value_counts(dropna =False))#value count of string object
data.describe()
data.columns
data.boxplot(column = 'MWG',by ='NWG')
data_new =data.head()

data_new
data.info()
melted = pd.melt(frame =data_new,id_vars= 'MWG_Level',value_vars=['MWG','NWG'])

melted