# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #visualization tool

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/heart.csv")
data.info()
data.corr()
plt.subplots(figsize=(10,10))

sns.heatmap(data.corr(), annot=True, linewidth=1, fmt='.2f')

#plt.show()
data.head()
data.columns
data.age.plot(kind='line', color='y', label='age', linewidth=1, alpha=.7, grid=True, linestyle=':')

data.cp.plot(kind='line', color='g', label='cp', linewidth=1,grid=True)

data.trestbps.plot(kind='line', color='r', label='trestbps', linewidth=1,grid=True)

plt.legend(loc='lower right')

plt.show()
data.plot(kind='scatter', color='r', x='age', y='cp', linewidth=1, alpha=.7, grid=True)

plt.xlabel('age')

plt.ylabel('cp')

plt.title('Age-Cp Scatter Plot') 

plt.show()
data.age.plot(kind='hist',bins=10, figsize=(5,5))

plt.show()

plt.clf()
dictionary={'banu':26,

          'başar':16,

           'zuhal':45}

print(dictionary.keys())

print(dictionary.values())

dictionary['banu']=25

dictionary['ersin']=50

#del dictionary['ersin']

print(dictionary.keys())

print(dictionary.values())

print(dictionary)

print('ersin' in dictionary)
series=data['age']

print(series)
dataframe=data[['age']]

print(dataframe)
x= data['age']>=72

data[x]
data.columns
data[np.logical_and(data['age']>=70,data['sex']==1)]
data[(data['age']>=70) & (data['sex']==1)]
lis=[1,2,3,4,5]

i=0

while lis[i]!=5:

    i+=1

    print(i)
for i in lis:

    print(i)
for i,value in enumerate(lis):

    print(i,':',value)
for key,value in dictionary.items():

    print(key,':',value)
for index,value in data[['age']][0:5].iterrows():

                                print(index,':',value)
def tuble_ex():

    t=(1,2,3)

    return t

a,b,c=tuble_ex()

print(a,b,c)
x=2

def f():

    x=3

    return x

print(f())

print(x)

x=3

def f(x):

    #x=4 bunu yazarsam 16,yazmazsam 9 döndürür.

    y=x**2

    return y

print(f(x))
import builtins

dir(builtins)
def square():

    def add():

        x=2

        y=3

        z=x+y

        return z

    y=add()**2

    return y

print(square())

    
def f(a,b=1,c=2):

    x=a+b+c

    return x

print(f(5))

print(f(5,4,3))
def f(*args):

    for i in args:

        print(i)

f(2)

print(" ")

f(1,2,3,4)
def f(**kwargs):

    for key,value in kwargs.items():

        print(key," ",value)

f(banu=25,başar=16)
square=lambda x:x**2

print(square(4))

top=lambda x,y,z:x+y+z

print(top(1,1,1))


top=map(lambda x:x**2,[1,2,3])

print(list(top))
name="banu"

it=iter(name)

print(next(it))

print("")

print(*it)

list1=[1,2,3]

list2=['a','b','c']

ziplist=zip(list1,list2)

zipli=list(ziplist)

print(zipli)
un_zip=zip(*zipli)

unlist1,unlist2=list(un_zip)

print(unlist1)

print(unlist2)

print(type(unlist2))

print(type(list(unlist1))) #unlist1 i tupledan listeye çevir tipini yaz
num1=[1,2,3]

num2=[i+1 for i in num1]

print(num2)
num1 = [5,10,15]

num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]

print(num2)
data.columns

data.info()
data=pd.read_csv("../input/heart.csv")

data.head()
data.tail()
data.shape
data.columns
print(data['sex'].value_counts(dropna =False)) 
data.describe()
data.boxplot(column='age',by='cp')
data_new=data.head()

data_new
melted=pd.melt(frame=data_new,id_vars='age',value_vars=['1','0'])

melted
melted.pivot(index='age', columns='variable',values='value')
data1=data.head()

data2=data.tail()

conc_data_row=pd.concat([data1,data2],axis=0,ignore_index=True)

conc_data_row
data1=data['age'].head()

data2=data['thalach'].tail()

conc_data_col=pd.concat([data1,data2],axis=1,ignore_index=True)

conc_data_col
data.dtypes
data['exang'] = data['exang'].astype('float64')

data['thal'] = data['thal'].astype('float')
data.dtypes
data.info()
data["age"].value_counts(dropna=False)
data1=data

data1["sex"].dropna(inplace=True)
assert 1==1
assert  data['sex'].notnull().all()
data["age"].fillna('empty',inplace = True)
data.columns
data1=data.loc[:,['age','cp','chol']]

data1.plot()
data1.plot(subplots=True)

plt.show()
data1.plot(kind="scatter",x="age",y="cp")

plt.show()
data1.plot(kind="hist",y="age",bins=50,range=(0,100),normed=True)

plt.show()
fig,axes=plt.subplots(nrows=2,ncols=1)

data1.plot(kind="hist",y="age",bins=50,range=(0,100),ax=axes[0])

data1.plot(kind="hist",y="age",bins=50,range=(0,100),ax=axes[1],cumulative=True)

plt.savefig('graph.png')

plt
time_list=["1993-02-10","1990-04-12"]

print(type(time_list[1]))

datetime_object=pd.to_datetime(time_list)

print(type(datetime_object))
import warnings

warnings.filterwarnings("ignore")
data2=data.head()

date_list=["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object=pd.to_datetime(date_list)

data2["date"]=datetime_object

data2=data2.set_index("date")

data2
print(data2.loc["1993-03-16"])

print(data2.loc["1992-03-10":"1993-03-16"])
data2.resample('A').mean()
data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")
data = pd.read_csv('../input/heart.csv')

data.head()
data["age"][1]

#data.age[1]
data.loc[1,["age"]]
data[["age","cp"]]
print(type(data["cp"]))     # series

print(type(data[["cp"]]))   # data frames
data.loc[:,"cp":"thal"]
data.loc[10:1:-1,"cp":"thal"] 
data.loc[1:5,"fbs":]
boolean=data.thalach<100

data[boolean]
filt1=data.age>70

filt2=data.sex==1

data[filt1&filt2]
data.age[data.sex==1]
def div(n):

    return n/2

data.age.apply(div)
data.sex.apply(lambda n:n*2)
data["total"]=data.cp+data.fbs

data.head()
print(data.index.name)

data.index.name="index_name"

data.head()
data.head()

data3=data.copy()

data3.index=range(10,313,1)

data3.head()
data1 = data.set_index(["sex","age"]) 

data1.head(10)
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df = pd.DataFrame(dic)

df
df.pivot(index="treatment",columns="gender",values="response")
df1=df.set_index(["treatment","gender"])

df1
df1.unstack(level=0)

df1.unstack(level=1)
df2=df1.swaplevel(0,1)

df2
df
# df.pivot(index="treatment",columns = "gender",values="response")

pd.melt(df,id_vars="treatment",value_vars=["age","response"])
df


df.groupby("treatment").mean()
df.groupby("treatment").age.max() 

df.groupby("treatment")[["age","response"]].min() 
df.info()