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
avocado=pd.read_csv("../input/avocado.csv")
avocado.info()
columns1=["#","Date","AveragePrice","Total Volume","4046","4225","4770","Total Bags","Small Bags","Large Bags","XLarge Bags","type","year","region"]
avocado.columns=columns1
avocado.info()
avocado.describe()
avocado.dtypes
avocado.columns
avocado.head()
avocado.tail()
avocado.corr()
f,ax=plt.subplots(figsize=(15,15))
sns.heatmap(avocado.corr() , annot=True ,linewidths=.5,fmt=".2f")
plt.title("Avocado Corralation Map")
plt.show()
#Line Plot

# Total number of avocados with PLU 4046 sold
# Total number of avocados with PLU 4225 sold

avocado["4046"].plot(kind="line",color="red",linestyle=":",label="4046",grid=True,alpha=0.5,figsize=(10,10))
avocado["4225"].plot(kind="line",color="green",linestyle=":",label="4225",grid=True,alpha=0.5,figsize=(10,10))
plt.legend()
plt.title("Data of 4046 and 4225 ")
plt.show()

#Scatter Plot

avocado.plot(kind="scatter",x="AveragePrice",y="Small Bags",color="g",grid=True,linestyle="-",figsize=(10,10))
plt.title("Average price of Small Bags")
plt.show()
#Histogram

avocado["AveragePrice"].plot(kind="hist",color="blue",bins=30,grid=True,alpha=0.65,label="Average Price",figsize=(10,10))
plt.legend()
plt.xlabel("Average Price")
plt.title("Average Price Distribution")
plt.show()
series=avocado["Total Bags"]
print(type(series))
df=avocado[["Total Bags"]]
print(type(df))
#1
filtre=avocado["Total Bags"]> 35500
avocado[filtre]
#2
avocado[avocado["region"]=="Atlanta"]
#1
avocado[np.logical_and(avocado["year"]==2015, avocado["Total Volume"]>10)]

#2
avocado[(avocado["type"]=="conventional") & (avocado["AveragePrice"]<0.6)]
def list1func():
    """ return defined list1 list """
    list1=["alex","hagi","maradona","sneijder"]
    return list1
a,h,m,s=list1func()
print(a,h,m,s)
#1
x=2
def func():
    x=3
    return x
print(x)        # global scope
print(func())   # local scope
#2
x=y=4          # global scope
def func2():
    x=y+1        #local scope
    return x           
print(x)
print(func2())
import builtins  #scopes provided by python
dir(builtins)
def func():
    """return value x*add"""
    def add():
        """add local variable """
        x=2
        y=8
        z=y+x
        return z
    return add()**2
print(func())
#default arguments

def func(x,y,z=3):      # default argument is overwritten
    """ return x+(y*z) """
    return z+(x*y)
print(func(2,1))
#flexible arguments

def func2(*args):
    for i in args:
        print(i)
func2(1,2,3,4,5)

#flexible arguments **kwargs --->> dictionary
def func3(**kwargs):
    for key, value in kwargs.items():
        print(key+":"+value)
        
func3(alex="Brazil",hagi="Romania")


def f(x):
    """  """
    return lambda y: y**x  
square=f(2)
cube=f(3)
print(cube(1))
print(square(2))
limit=avocado.AveragePrice.mean()
print(limit)
avocado["rating"]=["expensive" if i>limit else "cheap" for i in avocado.AveragePrice]
avocado.loc[0:25,["rating","AveragePrice"]]
#1
print(avocado["type"].value_counts(dropna=False))  #Shows the number of different avocado types
#2
print(avocado["year"].value_counts(dropna=False)) 
print(avocado.boxplot(column="AveragePrice",by="year"))
#1
newData=avocado.head()
melted=pd.melt(frame=newData,id_vars="type",value_vars=["Small Bags","Total Bags"])
melted
data1=avocado[avocado["Total Bags"]>15000000]
data2=avocado[avocado["Small Bags"]>10000000]
#print(data1,data2)
concData=pd.concat([data1,data2],axis=0,ignore_index=True)
concData
#print(data1,data2,concData)
avocado.dtypes
avocado["year"]=avocado["year"].astype("float")
avocado.region=avocado.region.astype("category")
avocado.dtypes
avocado.info() 
avocado["XLarge Bags"].value_counts(dropna =False) #no null value
assert  avocado["XLarge Bags"].notnull().all()   #return empty
#Edit columns name 
#avocado.columns = avocado.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
#avocado.columns
data1=avocado.loc[:,["Small Bags","AveragePrice","Total Bags"]]
data1.plot()
#1 subplots
data1.plot(subplots=True)
plt.show()
#2 scatter plot
data1.plot(kind="scatter",x="AveragePrice",y="Small Bags")
plt.show()
#3 hist plot
data1.plot(kind="hist",y="AveragePrice",bins=50,range=(0,1))
#histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "AveragePrice",bins = 50,range= (0,1),normed = True,ax = axes[0])
data1.plot(kind = "hist",y = "AveragePrice",bins = 50,range= (0,1),normed = True,ax = axes[1],cumulative=True)
plt.savefig('graph.png')
plt
avocado.describe()
time_list = ["1992-03-08","1992-04-12"]
print(type(time_list[1])) # As you can see date is string
# however we want it to be datetime object
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))
import warnings
warnings.filterwarnings("ignore")
data2 = avocado.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object
# lets make date as index
data2= data2.set_index("date")
data2
# Now we can select according to our date index
print(data2.loc["1993-03-16"])
print(data2.loc["1992-03-10":"1993-03-16"])
data2.resample("A").mean()
data2.resample("M").mean()
data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")
avocado=avocado.set_index("#")
avocado.head()
avocado["AveragePrice"][1]
avocado.AveragePrice[1]
avocado.loc[1,["AveragePrice"]]
avocado[["AveragePrice","Total Bags"]]
print(type(avocado["Total Bags"]))  #series
print(type(avocado[["Total Bags"]]))  #data frames
avocado.loc[::,"AveragePrice":"Total Bags"]
avocado.loc[::-1,"AveragePrice":"Total Bags"]
avocado.loc[:,"Large Bags":]
#1
filter1=avocado["AveragePrice"]>3
avocado[filter1]
#2
filter2=avocado["AveragePrice"]>3
filter3=avocado["year"]==2017
avocado[filter2 & filter3]
#3
avocado.AveragePrice[avocado.AveragePrice>3]
def div(n):
    return n*2

avocado.AveragePrice.apply(div)
avocado.AveragePrice.apply(lambda n:n*2)
avocado["Total"]=avocado["Total Bags"]+avocado["Total Volume"]
avocado.head()
print(avocado.index.name) #output "#"
avocado.index.name="index_name"
avocado.head()
data=avocado.copy()
data.index=range(0,18249,1)
data.head(50)
data2=avocado.set_index(["region","type"])
data2.head(500)
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
dframe = pd.DataFrame(dic)
dframe
dframe.pivot(index="gender",columns = "treatment",values="response")
df1 = dframe.set_index(["treatment","gender"])
df1
df1.unstack(level=0)
df1.unstack(level=1)
df2 = df1.swaplevel(0,1)
df2
#reverse of pivoting
pd.melt(dframe,id_vars="treatment",value_vars=["response","age"])
dframe
dframe.groupby("treatment").mean()
dframe.groupby("treatment").age.max()
dframe.groupby("response").age.max()
dframe.groupby("treatment")[["age","response"]].min()