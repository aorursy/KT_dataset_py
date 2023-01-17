# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#We read data here

data = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")
#We examine our data, example:

#data.info()

#data.columns

data.head()
#I use filtre here beacuse id 10472 data rating is 19.0 and that is impossible

filtrete = data["Rating"]<=5

data[filtrete].Rating.plot(kind="line",color="blue",grid=True,label="Rating",linewidth=0.5,alpha=9,linestyle="-")

plt.xlabel("ID")

plt.ylabel("Rating")

plt.title("Rating Line Graph")

plt.show()
#I can't scatter graph because our data has 1 datatype is numberic
data[filtrete].Rating.plot(kind="hist",bins=40,figsize=(13,13),grid=True,color="red")

plt.xlabel("Rating")

plt.ylabel("Number")

plt.title("Rating Histogram Graph")

plt.legend(loc="best")

plt.show()
y = data[np.logical_and(data["Category"]=="ART_AND_DESIGN",data["Type"]=="Free")]

print(len(y))
dictionary = {"Azerbaycan":"Bakü","Türkiye":"Ankara","Rusya":"Berlin"}

for key,value in dictionary.items():

    print(key," : ",value)

for index,value in data[["Rating"]][0:10].iterrows():

    print(index," : ",value)
#plt.clf()

data[filtrete].Rating.plot(kind="hist",bins=40,figsize=(13,13),grid=True,color="red")

plt.xlabel("Rating")

plt.ylabel("Number")

plt.legend(loc="best")

plt.clf()
#User defined function and scope

x = 12

y = 9            #Global scope

def list1():

    x = 5        #Local scope

    y = 7        #If we delete y in our function, then be y=9

    return [x,y]

print([x,y])

print(list1())
#Nested Functions

def trigon():

    def value():

        edge = 10

        high = 6

        return edge*high

    return value()/2

trigon()
#Default arguments

def default(name="user"):

    return name

print("Hello",default())

print("Hello "+str(default("admin")))
#Flexible arguments

total = 0

def flexible(*args):

    global total

    for i in args:

        total = total + i

    return total

print(flexible(1,2,3,4))
#Flexible arguments continue

def func(**kwargs): #kwargs is dictionary

    for key,value in kwargs.items():

        print(key," = ",value)

func(German ="Berlin",Russian="Moskova")
#Lambda

trigon = lambda x,y: (x*y)/2

print(trigon(2,2))

#User function

def trigon2(x,y):

    return (x*y)/2

print(trigon2(2,2))

#As you see both is same but one is than short
#Anonymous

numbers = {1,2,3}

y = list(map(lambda x:x*2,numbers))

print(y)
#Iterable

isim = "ali"

ourIter = iter(isim)

print(next(ourIter))

print(*ourIter)
#Zip

list1 = [0,1,2]

list2 = ["x","y","z"]

z = zip(list1,list2)

print(z)

z_list = list(z)

print(z_list)

#Unzip

un_zip = zip(*z_list)

un_list1,un_list2 = un_zip

print(un_list1,"\n",un_list2)

#un_list1 = list(un_list1)

#print(un_list1)
#List comprehension

numbers1 = "1234"

numbers2 = [int(i)**2 for i in numbers1]

print(numbers2)
numbers = [3,5,10]

nums = [i-i if i<5 else i*2 if i == 5 else i**2 for i in numbers]

print(nums)
 #List comprehension in our data

filtrete = data["Rating"]<=5

average = sum(data[filtrete].Rating)/len(data[filtrete].Rating)#Some codes is faulty why I used filter

print(round(average,2))

for i in range(10841):

    try:

        data["RatingAverage"]=["high" if i>average else "low" for i in data.Rating]

        #You can asking we why don't filter because if use filter we outrun the index

    except:

        data["RatingAverage"]=["Not"]

data.loc[:10,["RatingAverage","Rating"]]
#We can again:

#columns,head(),tail(),info(),shape etc.
print(data.RatingAverage.value_counts(dropna =False))
data[filtrete].describe()
values = [1,7,8,6,71,9,8]

values.sort()# sort() function is sort to values

print(values)

#count is 7

#mean is 15,7

#median is 8

#Q1 or first quaile is 6,5

#Q3 or third quaile is 8,5
data.boxplot(column="Rating",figsize=(16.5,16.5))

plt.show()
new_data= data.tail()

new_data
melted = pd.melt(frame=new_data,id_vars="App",value_vars=["Genres","Size"])

melted
melted.pivot(index = "App",columns="variable",values="value")
#Row



data1 = data.head()

data2 = data.tail()

data3 = pd.concat([data1,data2],axis=0,ignore_index=True)

data3
#Columns



apps = data["App"].head()

reviews = data["Reviews"].head()

currentVer = data["Current Ver"].head()

data3 = pd.concat([apps,reviews,currentVer],axis=1)

data3
a = data.dtypes

a
data["Rating"] = data["Rating"].astype("float")# We can't use astype that data

data.dtypes
data.info()
data.Rating.value_counts(dropna=False)
data1 = data.copy()

data1.Rating.dropna(inplace=True)
data.Rating.value_counts(dropna=False)
assert 2**4==4**2 #We  use assert for test our operation
assert data1.Rating.notnull().all # returns nothing
data1.Rating.fillna("Not",inplace=True)
assert data1.Rating.notnull().all
assert data1.columns[0] == "App"

assert data1.Reviews.dtype == "object"
#data frames from dictionary

students = ["Mansur","Jale"]

grades = [100,90]

label = ["Name","Grade"]

col = [students,grades]

zipped = list(zip(label,col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
#Add new columns

df["Average"] = ["A+","A"]

df
#Broadcast

df["Status"] = "Passed"

df
data1 = data.loc[:,["Rating","Rating"]]# We have 1 numeric dataype :(

data1.plot()
data1.plot(subplots=True)

plt.show()
data1.plot(kind="scatter",x="Rating",y="Rating")
data1.plot(kind="hist",y="Rating",bins=50,range=(4,5))#I can't use normed
fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind="hist",y="Rating",bins = 50,range= (0,5),ax=axes[0])

data1.plot(kind="hist",y="Rating",bins = 50,range= (0,5),ax=axes[1],cumulative = True)

plt.savefig('graph.png')

plt
time_list = ["2005/03/28","2005/01/18"]

datetime_object = pd.to_datetime(time_list)

datetime_object
import warnings

warnings.filterwarnings("ignore") #Close warning



data2 = data.head()

date_list = ["1923-10-23","1923-10-24","1923-11-23","1923-11-24","1925-01-18"]

datetime_object = pd.to_datetime(date_list)#Our date_list' change datetime

data2["Date"] = datetime_object

data2 = data2.set_index(datetime_object)#Set index to datetime

data2
print(data2.loc["1925-01-18"])

print(data2.loc["1923-11-23":"1925-01-18"])
#Resampling

data2.resample("A").mean()#Years
data2.resample("M").mean()#Months
data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")
data = pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv")

data.head()
assert data["Size"][0]==data.Size[0]

data["Size"][0]# indexing using square brackets
data.Size[0]# using column attribute and row label
data.loc[0,["Size"]]# using loc accessor
data[["Category","Type"]]
print(type(data["Rating"]))

print(type(data[["Rating"]]))
data.loc[0:10,"Rating":"Size"]
#data.loc[1:10:-1,"Rating":"Size"] I do not why not reverse
data.loc[0:,"Genres":]
filt = data.Rating<2

data[filt]
first_filt = data.Rating<2

other_filt = data.Category=="FAMILY"

data[first_filt & other_filt]
data.Reviews[data.Rating==1]
def funt(h):

    return h/2

data["Rating"].apply(funt)
data.Rating.apply(lambda h: h/2)
data["Rating/10"] = data.Rating + data.Rating

data.head()
data.index.name = "Index"

data.head()
data3 = data.copy()

data3.index = range(100,10941,1)

data3.tail()
data = pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv") # We refresh our data
data1 = data.set_index(["Category","Genres"])

data1.head()
dict1 = {"City":["A","B","A","B"],"GradeType":["L","L","U","U"],"Population":[25,75,60,90],"Grade":[50,75,100,45]}

df = pd.DataFrame(dict1)

df
#Pivot

df.pivot(index="City",columns="GradeType",values="Grade")
df1 = df.set_index(["City","GradeType"])

df1
df1.unstack(level=0)
df1.unstack(level=1)
#df1

df2 = df1.swaplevel(0,1)

df2
pd.melt(df,id_vars="GradeType",value_vars=["Grade","Population"])
df.groupby("GradeType").mean()
df.groupby("GradeType").Population.mean()
df.groupby("GradeType")[["Population","Grade"]].mean()