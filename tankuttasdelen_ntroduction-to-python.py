# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  #visualization tool

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Iris.csv')
data.info()
data.head(20)
data.columns
data.shape
data.corr()
f,ax = plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(), annot = True, linewidths=.5 , fmt='.1f', ax=ax)

plt.show()
#line plot

data.SepalLengthCm.plot(kind= 'line', color='g', label='sepal',linewidth=1, alpha=.5 , grid=True, linestyle = ':') # g: green , b: blue , alpha: transparency level 

data.PetalLengthCm.plot(kind='line',color='b', label='petal',linewidth=1, alpha=.5, grid=True, linestyle = '-')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.show()
def fx():

    t=(1,2,3)

    return t

a = fx()

print(a)
x=3

def bro():

    x=21

    return x

print(x)

x1=bro()

print('x1:',x1,'_dir akıllı ol')
x=3

def bro():

    y=x*3

    return y,x

bro()
import builtins

dir(builtins)
def square():

    

    def add():

        x=3

        y=2

        z=x+y

        return z

    return add()**2

print(square())   
def f(a,b=1,c=2):

    x=a+b+c

    return x

print(f(5))

print(f(5,3,4))
def f(**amca):

    for key,value in amca.items():

        print(key," ",value)

f(country='İngland',capital='londra',populatin=84000)

def f(*amca):

    for i in amca:

        i=i+1

        print(i)

f(1,2,3,4)
kare_alma= lambda x: x**2

print(kare_alma(3))

average= lambda x,y,z: (x+y+z)/3

print(average(4,6,8))
sayı_listesi=[2,5,90]

ka=map(lambda x: x**2,sayı_listesi)

print(list(ka))
# zip method

listem1=[4,2,5,0]

listem2=[16,4,25,81]

y=zip(listem1,listem2)

print(y)

y_listem=list(y)

print(y_listem)



# unzip method



listem=zip(*y_listem)

lis1,lis2=list(listem)

print(lis1)

print(lis2)

print(type(list(lis1)))
# conditionals on iterable

num1=[1,20,3]

num2=[i+1 for i in num1]

print(num1,num2)

num3=[i*10 if i==1 else i*5 if i>3 else i**2 for i in num1]

print(num3)
data.head()
data.tail()

print(data['SepalWidthCm'].value_counts(dropna=False))
data.describe()
data.boxplot(column='SepalLengthCm',by='Species')

plt.show()
data.columns
data_new=data.head()

melted=pd.melt(frame=data_new,id_vars='Species', value_vars=['SepalLengthCm','PetalLengthCm'])

melted              
data1=data.head()

data2=data.tail()

concatenating_data=pd.concat([data1,data2],axis=0,ignore_index=True)

concatenating_data
data1=data['SepalLengthCm'].head()

data2=data['PetalLengthCm'].head()

cont_data=pd.concat([data1,data2],axis=1)

cont_data

data.dtypes
data['Id']=data['Id'].astype('float')
data.dtypes
data.tail(20)
data.head(20)
data['Id']=data['Id'].astype('int')
data.dtypes
data.head()
data.info()
data["Species"].value_counts(dropna=True)
assert data.columns[1]== 'SepalLengthCm'
country=["spain","italian",]

population=["15" , "9",]

birinci=["nadal","jokovic"]

list_label=["country","population","birinci"]



list_col=[country,population,birinci]



zipped=list(zip(list_label,list_col))

data_dict=dict(zipped)

df=pd.DataFrame(data_dict)

df
df["ekonomi"]=["zayıf","orta"]

df
df["adam_sayısı"]=0

df
data1=data.loc[:,["SepalLengthCm","SepalWidthCm","PetalLengthCm"]]

data1.plot()

plt.show()
data.columns
data1.plot(subplots=True)

plt.show()
data1.plot(kind="scatter",x="SepalLengthCm",y="PetalLengthCm")

plt.show()
data.plot(kind="hist",y="SepalLengthCm",bins=50)

plt.show()
data1.plot(kind="hist",y="SepalLengthCm",bins=50,normed=True)

plt.show()
data2=data.head()

data_list=["1993-06-29","1993-07-29","1993-08-29","1994-10-29","1994-10-28"]

data_time_object=pd.to_datetime(data_list)

data2["date"]=data_time_object

data2=data2.set_index("date")

data2
print(data2.loc["1993-06-29"])

print(data2.loc["1993-06-29":"1993-08-29"])
data2.resample('A').mean()
data2.resample('M').mean()
data2.resample('M').mean().interpolate("linear")
data2.resample('M').mean()
data2.head()
data3=data.head()

data3
data3["SepalLengthCm"][0]
data[["SepalLengthCm", "Id"]]
data.loc[0:20,"SepalLengthCm":"PetalLengthCm"]
data.loc[20:0:-1,"SepalLengthCm":"PetalLengthCm"]
data.loc[0:10,"PetalLengthCm":]
data[data.Id > 145]
firs_filter=data.Id >140

second_filter=data.SepalLengthCm >6

data[firs_filter & second_filter]
data[(data.Id >140) & (data.SepalLengthCm >6.5) ]
data.Id[data.SepalLengthCm>6.7]
data3

def div(n):

    return n/2

data3.PetalLengthCm.apply(div)
data3.PetalLengthCm.apply(lambda n: n*2)

data4=data.head(20)

data4
data4["total_LengthCm"]=data4.SepalLengthCm + data4.PetalLengthCm

data4
data.index.name="index_name"

print(data.index.name)
data12=data.copy()

data12.index=range(100,400,2)

data12.head()
data=pd.read_csv('../input/Iris.csv')

data=data.set_index("Id")

data.head()
dic={"yaş":[11,15,56,5],"cinsiyet":["M","M","F","M"],"tedavi":["ilaç","fzt","ilaç","ameliyat"],"sonuc":["iyi","devam","kötü","devam"]}

data_frame=pd.DataFrame(dic)

data_frame["index"]=[1,2,3,4]

data_frame=data_frame.set_index("index")

data_frame