# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")

data
data.info()
data.describe()
data.columns
data.corr()
f,ax=plt.subplots(figsize=(8,8))

sns.heatmap(data.corr(),annot=True,lw=.5,fmt=".1f",ax=ax)

plt.show()
data.head(3)
data["math score"].plot(kind="line",color="b",label="math score",ls="-",lw=.7,alpha=.6,grid=True,figsize=(10,10))

data["writing score"].plot(kind="line",color="r",label="writing score",ls="-",lw=.5,alpha=.5,grid=True)

data["reading score"].plot(kind="line",color="g",label="writing score",ls="-",lw=.5,alpha=.4,grid=True)

plt.xlabel("x axis",color="m")

plt.ylabel("y axis",color="m")

plt.show()
data.plot(kind="scatter",x="reading score",y="writing score",color="orange",alpha=.6,figsize=(9,9),grid=True)

plt.xlabel("reading score",color="k")

plt.ylabel("writing score",color="k")

plt.title("Reading vs. Writing Score")

plt.show()
plt.subplots(figsize=(10,10))

plt.scatter(data["reading score"],data["writing score"],color="c",alpha=.5)

plt.grid(True)
data["reading score"].plot(kind="hist",color="m",alpha=.5,grid=True,figsize=(8,8))
plt.hist(data["reading score"])
series=data["reading score"]

series
data_frame=data[["reading score"]]

data_frame
#1.

x=data["math score"]>97

data[x]
#2.

data[np.logical_and(data["math score"]>97,data["reading score"]>98)]
#3.

data[(data["reading score"]>98)&(data["math score"]>97)]
#4.

data[(data["gender"]=="female")&(data["test preparation course"]=="completed")&(data["writing score"]>99)&(data["math score"]>95)&(data["lunch"]=="standard")]
#5.

first_filter=data["math score"]>97

second_filter=data["reading score"]>97

data[first_filter & second_filter]
#6.

data["reading score"][data["math score"]>97]
#While loop

i=0

while i!= 8:

    print(i)

    i+=2

print(i,"is equal to 8")
#for loop

#1)lists:

liste1=[5,1,8,-3,0,4,13]

for index,value in enumerate(liste1):

    print("index:",index,", value:",value)
#2)dictionaries:

dictionary={"turkey":"ankara",

            "usa"   :"washington dc",

            "china" :"pekin"}

for key,value in dictionary.items():

    print("key:",key,", value:",value)
#3)pandas(dataframes):

for index,value in data[["writing score"]][0:4].iterrows():

    print(index,value)
# *args

def f(*args):

    for i in args:

        print(i)

f(1)

f("")

f(1,2,3,7)
f("asdasdasdas",4545)
# **kwargs

def f(**kwargs):

    """print key and value of dictionary"""

    for key,value in kwargs.items():

        print("key:",key,"value:",value)

f(country = 'spain', capital = 'madrid', population = 123456)
total=lambda x,y,z,t : x*y+z*t

total(3,-1,4,7)
liste2=[1,2,3,4,5]

x=map(lambda x : x**(1/2),liste2)

list(x)
name="marcelo"

it=iter(name)
next(it)
next(it)
next(it)
print(*it)
list1=[1,3,5,6,0]

list2=[40,56,8,-17,8]

z=zip(list1,list2)

a=list(z)

a
un_zip=zip(*a)

unlist1,unlist2=list(un_zip) #burada list yada tuple diyebiliriz.

print(unlist1)               #ikisinde de tuple yapar.

print(unlist2)

print(type(unlist2))
print(list(unlist1))

print(type(list(unlist1)))
m=[1,2,3,5,7,11,13]

n=[i**2 for i in m]

n
p=[20,35,40]

q=[i**2 if i%10==0 else i/5 for i in p]

q
data.head(3)
print(data["race/ethnicity"].value_counts(dropna=False))
print(data["gender"].value_counts(dropna=False))
print(data["parental level of education"].value_counts(dropna=False))
print(data["lunch"].value_counts(dropna=False))
print(data["test preparation course"].value_counts(dropna=False))
print(data["math score"].value_counts(dropna=False))
x=sum(data["math score"])/len(data["math score"])

x
data.head(3)
data.boxplot(column="math score",by="gender",figsize=(8,8),grid=True)
data.boxplot(column="math score",by="gender",figsize=(8,8),grid=True,notch=True)
data.boxplot(column="math score",by="gender",figsize=(8,8),grid=True,notch=True,patch_artist=True)
plt.subplots(figsize=(10,10))

data2 = [np.random.normal(0, std, 1000) for std in range(1, 6)]

box = plt.boxplot(data2, notch=True, patch_artist=True)

colors = ['cyan', 'lightblue', 'lightgreen', 'tan', 'pink']

for patch, color in zip(box['boxes'], colors):

    patch.set_facecolor(color)

plt.show()
data_new=data.head()

data_new
data["math score"]=data["math score"].astype(int)

data["reading score"]=data["reading score"].astype(int)

data["writing score"]=data["writing score"].astype(int)
melted=pd.melt(frame=data_new,id_vars="race/ethnicity", value_vars=["math score","reading score","writing score"])

melted
#melted.pivot(index="race/ethnicity",columns="variable",values="value")
data3=data.head()

data4=data.tail()

concat_row=pd.concat([data3,data4],axis=0,ignore_index=True)

concat_row
concat_row2=pd.concat([data3,data4],axis=0)

concat_row2
concat_row3=pd.concat([data3,data4],axis=1,ignore_index=True)

concat_row3
data["math score"]=data["math score"].astype("int")
data.dtypes
data5=data.copy()

data5["parental level of education"].dropna(False)
data["test preparation course"].fillna("empty",inplace=True)

# bu columndaki boş olanları empty yazısı ile doldurduk.
data.head(3)
race=["group A","group B","group C"]

edu=["bachelor's degree","some college","master's degree"]

gender=["male","female","empty"]

list_label=["race","edu","gender"]

list_col=[race,edu,gender]

zipped=list(zip(list_label,list_col))

data_dict=dict(zipped)

df=pd.DataFrame(data_dict)

df
data_dict
df["success"]=["yes","no","no"]
df
df["country"]="US"
df
data6=data.loc[:,["math score","reading score","writing score"]]

data6.plot()
import warnings

warnings.filterwarnings("ignore")
data6.plot(subplots=True)

plt.legend(loc="upper left")

plt.show()
data6.plot(kind="scatter",x="reading score",y="writing score")

plt.show()
data6.plot(kind="hist")
data6.plot(kind="hist",x="math score")
data6.plot(kind="hist",y=["math score","reading score"])
data6.plot(kind="hist",y="math score",bins=40,range=(0,120))
data6.plot(kind="hist",y="math score",bins=40,range=(0,120),rwidth=.2,orientation="horizontal")
data6.plot(kind="hist",y="math score",bins=40,range=(0,120),rwidth=.7,orientation="vertical")
fig,axes=plt.subplots(nrows=2,ncols=1)

data6.plot(kind="hist",y="math score",bins=50,ax=axes[0]) #noncumulative

data6.plot(kind="hist",y="math score",bins=50,ax=axes[1],cumulative=True)  #cumulative

plt.savefig("cumulative.png")

plt.show()
fig,axes=plt.subplots(nrows=2,ncols=3,figsize=(12,9))



data6.plot(kind="hist",y="math score",bins=50,ax=axes[0][0]) #noncumulative

data6.plot(kind="hist",y="math score",bins=50,ax=axes[1][0],cumulative=True)  #cumulative

data6.plot(kind="hist",y="reading score",bins=50,ax=axes[0][1])

data6.plot(kind="hist",y="reading score",bins=50,ax=axes[1][1],cumulative=True)

data6.plot(kind="hist",y="writing score",bins=50,ax=axes[0][2])

data6.plot(kind="hist",y="writing score",bins=50,ax=axes[1][2],cumulative=True)

plt.savefig("cumulative.png")

plt.show()
data7=data.tail()

data7
data_list=["2020-10-1","2018-8-3","2018-2-5","2018-2-5","2016-2-7"]

datatime=pd.to_datetime(data_list)

data7["date"]=datatime

data7
data7=data7.set_index("date")

data7
print(data7.loc["2018-02-05	"])
print(data7.loc["2020-10-01":"2018-02-05"])
data7.resample("A").mean()
data7.resample("M").mean()
data7.resample("M").first().interpolate("linear")
data7.resample("M").mean().interpolate("linear")
data.head()
data.loc[1,["parental level of education"]]
data[["gender","math score"]]
type(data["gender"])
type(data[["gender"]])
data.loc[0:10,"gender":"lunch"]
data.loc[0:10,["gender","lunch"]]
data.loc[10:0:-1,["gender","lunch"]]
data.loc[2:10,"lunch":]
#1st

#def and apply

def div(n):

    return n/10

data["writing score"].apply(div)
#2nd

#lambda and apply

data["writing score"].apply(lambda n : n/10)
#3rd

# Defining column using other columns

data["language_score"]=data["reading score"]+data["writing score"]

data.head()
data.language_score.head()
#1st

#Learning Index's name

print(data.index.name)
#2nd

#Changing index's name

data.index.name="index"

data.head(3)
#3rd

data8=data.copy()

data8.index=range(100,2100,2)

data8
data9=data.set_index(["gender","lunch"])

data9.head(20)
#stacking

data10=data.head(10).set_index(["gender","math score"])

data10
#unstacking

data10.unstack()

#default => level 0, 1st index column is deleted
data10.unstack(level=0)

#level1, 0th index column is deleted
data10.unstack(level=1)

#level 0, 1st index column is deleted
# changing inner and outer level index position

data11=data10.swaplevel(0,1)

data11
data12=data10.swaplevel(1,0)

data12
data9
pd.melt(data9,id_vars="gender",value_vars=["language_score","math score"])
data9.groupby("gender").mean()
data9.groupby("gender")["language_score"].max()
data9.groupby("gender").max()
data9.groupby("gender")[["language_score","math score"]].min()
data9.groupby("gender").min()
data.head()
data["parental level of education"].value_counts()
data["parental level of education"].unique()