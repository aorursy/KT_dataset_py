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
bitcoin_data =pd.read_csv("../input/bitflyerJPY_1-min_data_2017-07-04_to_2018-06-27.csv")
bitcoin_data.info() #data information
bitcoin_data.describe() #numeric value
bitcoin_data.columns
bitcoin_data.head() #default first 5
bitcoin_data.tail()
bitcoin_data.corr()
#correlation map 
f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(bitcoin_data.corr(),annot=True,linewidths=4,fmt=".1f",ax=ax)
plt.show()
bitcoin_data.isnull
#line plot
bitcoin_data.Low.plot(kind="line",color='r',label="low",linewidth=1.5,alpha=0.5,grid=True,linestyle=":")
bitcoin_data.High.plot(kind="line",color="b",label="high",linewidth=1,alpha=0.5,grid=True,linestyle="-")
plt.legend(loc = "upper right")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("line plot")
plt.show()
#scatter
# x= low, y=high
bitcoin_data.plot(kind="scatter",x="Low",y="High",alpha=0.5,color="r")
plt.xlabel("low")
plt.ylabel("high")
plt.show()
# Histogram
# bins = number of bar in figure
# bins = number of bar in figure
#data.Speed.plot(kind = 'hist',bins = 50,figsize = (12,12))
bitcoin_data.Close.plot(kind = "hist",bins = 40,figsize=(15,15))
plt.show()
#clf clean plot 
bitcoin_data.High.plot(kind="hist",bins = 40)
plt.clf() 
#we cannot plot

series =bitcoin_data["High"] #data["High"] = series
print(type(series))
data_frame = bitcoin_data[["High"]] #data[["High"]] = data frame
print(type(data_frame))
x =bitcoin_data["High"]>300000
print(x)
bitcoin_data[np.logical_and(bitcoin_data["Open"]>290000 ,bitcoin_data["Close"]<300000)]
lis=[1,2,3,4,5]
for i in lis:
     print('i is :',i)
print('')   
# Enumerate index and value of list
# index : value = 0:1, 1:2, 2:3, 3:4, 4:5
for index,value in enumerate(lis):
    print(index," :",value)
print('') 
# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.
dicto={'Turkey':'Ankara','spain':'madrid','canada':'toronto'}
for key,value in dicto.items():
        print(key,":",value)
print("")      
# For pandas we can achieve index and value
for index,value in bitcoin_data[['High']][0:1].iterrows():  
    print(key,":",value)
    
#tuble: sequence of immutable python objects.
#cant modify values
#tuple uses paranthesis like tuple = (1,2,3)
#unpack tuple into several variables like a,b,c = tuple

def tuple_x():
    """ return defined t tuble"""
    t =(1,2,3)
    return t 
a,b,c= tuple_x()
print(a,b,c)

#Scope
#What we need to know about scope:
# global: defined main body in script
# local: defined in a function
#* built in scope: names in predefined built in scope module such as print, len
x =2 
def f():
    x =3
    return x
print(x) #x is global
print(f()) #x is local
# What if there is no local scope
x=4
def f ():
    y=2*x  #x is not local spoce 
    return y 
print(f())   # it uses global scope x
# First local scopesearched, then global scope searched, if two of them cannot be found lastly built in scope searched.
# How can we learn what is built in scope
import builtins
dir(builtins)

#nested function: inside to funtion 
def squared():
    """return square of value """
    def add():
        """ return add to  2  local value"""
        x=2
        z=4
        y=x+z
        return y 
    return add()**2
print(squared())
#default argment
def f(a,b=2,c=3):
    y = a + b + c
    return y 

print(f(5))
    # what if we want to change default arguments
print(f(5,4,6))
#flexible argument *args
def f(*args) :
    for i in args :
        print(i)
f(1)
print("")
f(1,23,5)

#flexible argument **kwargs
def f(**kwargs):
    """return key,value on dictionary"""
    for index,value in kwargs.items():
        print(index,":",value)
f(country = "spain",capital ="madrid",population = 1478522)
#lambda funtion :faster function
square = lambda x :x**2
print(square(5))
total=lambda x,y,z : x+y+z
print(total(1,2,3))
# anonymous function : it like lambda but can be more then one argument
number=[1,2,3,4]
y =map(lambda x :x**3,number)
print(list(y))

#iteration
name="rihanna"
it =iter(name)
print(next(it)) #first /next iteration
print(*it) #remaning iteration
#zip():zip list
list1=[1,2,3,4]
list2=[7,8,9,6]
z=zip(list1,list2) 
print(z)
z_list=list(z) #convert z to list(z_list)
print(z_list)
un_zip =zip(*z_list)
ulist1,ulist2=list(un_zip) #return to unzip yo tuble
print(ulist1)
print(ulist2)
print(type(ulist1))
#list comperhension:collapse for loops for building lists into a single line 
#ex: num1 = [1,2,3] and we want to make it num2 = [2,3,4]. This can be done with for loop. 
#However it is unnecessarily long. We can make it one line code that is list comprehension
num1=[1,2,3]
num2=[ i+1  for i in num1]
print(num2)

#[i + 1 for i in num1 ]: list of comprehension
#i +1: list comprehension syntax
#for i in num1: for loop syntax
#i: iterator
#num1: iterable object

# Conditionals on iterable
num1=[7,5,6]
num2 = [i**2 if i==5  else i-7 if i==7 else i+1 for i in num1]
print (num2)
nu=[9,10,4]
nu2 =[  i+5 if i<5 else i+1 if i<10  else i  for i in nu]
print(nu2)
# lets classify bitcoin  whether they have high or low speed. Our threshold is average speed
threshold=sum(bitcoin_data.Open)/len(bitcoin_data.Open)
bitcoin_data["Open_Level"] = [" high" if i >threshold else "low" for i in bitcoin_data.Open]
bitcoin_data.loc[:10,["Open_Level","Open"]]  #we will learn loc more detailed later

data=pd.read_csv("../input/bitflyerJPY_1-min_data_2017-07-04_to_2018-06-27.csv")
data.head() #first 5 rows
data.tail() #last 5 rows
data.columns


data.shape #shape gives number of rows and columns in a tuple
data.info

print(data.Open.value_counts(dropna=True))
data.describe()
data.boxplot(column="Open",by="High")
data_new =data.head()
data_new
# lets melt
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted=pd.melt(frame=data_new,id_vars="Close",value_vars=["High","Low"])
melted
# Index is name
# I want to make that columns are variable
# Finally values in columns are value
melted.pivot(index='Close',columns= 'variable',values= 'value')
#firstl create 2 frame 
data1=data.head()
data2=data.tail()
cont_data_row=pd.concat([data1,data2],axis=0,ignore_index=True) #axis=0 add datadrame in  rows
cont_data_row


data1=data["High"].head()
data2=data["Low"].head()
cont_data_col=pd.concat([data1,data2],axis=1,ignore_index=True)
cont_data_col
data.dtypes
# lets convert int to categorical and  float.
data['Timestamp']=data['Timestamp'].astype("category")
data["Volume_(BTC)"]=data["Volume_(BTC)"].astype("int")
data.dtypes
#as you can Timestamp convert int to categorry
#Volume_BTc convert from float to int
data.info()
data["Timestamp"].value_counts(dropna=False)
data1=data
data1["Open"].dropna(inplace=True) # inplace = True means we do not assign it to new variable. Changes automatically assigned to data
# Lets check with assert statement
# Assert statement:
assert 1==1 # return nothing because it is true
# In order to run all code, we need to make this line comment
#assert 1==2 # return error because it is false
assert data['Open'].notnull().all() #returns nothingg becasue we drop non value
data['Open'].fillna('empty',inplace=True)
assert data['Open'].notnull().all() #returns nothing beacuse we do not not have nan values


# # With assert statement we can check a lot of thing. For example
# assert data.columns[1] == 'Open'
# assert data.Open.dtypes == np.int


#data frame from dictionary
country=["spain","france"]
population=["12","15"]
list_label=["country","population"]
list_col=[country,population]
zipped=list(zip(list_label,list_col))
data_dict=dict(zipped)
df = pd.DataFrame(data_dict)
df

#add news columns
df["capital"]=["madrid","paris"]
df
#broadcasting
df["incame"]=0 #broadcast entire colunmn
df
#ploting all data
data1=data.loc[1:,["Open","Low","High"]]
data1.plot()
#it is confused
#subplot
data1.plot(subplots=True)
plt.show()
#scater
data1.plot(kind="scatter",x="Low",y="High")
plt.show()
#hist
data1.plot(kind="hist",y="Low",bins=60,range=(2500,3000000),normed= True)
# histogram subplot with non cumulative and cumulative
fig,axes=plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist", y="Low",bins=50 ,range=(25000,3000000),normed=True, ax=axes[0])
data1.plot(kind ="hist",y="Low",bins=50, range=(25000,3000000),normed=True,ax=axes[0],cumulative=True)
plt.savefig("graph.png")
plt.show()

data.describe()
time_list=["2018-02-10","2018-03-04"]
print(type(time_list)) #let see string
# however we want it to be datetime object
datetime_object=pd.to_datetime(time_list)
print(type(datetime_object))
# close warning
import warnings
warnings.filterwarnings("ignore")

# In order to practice lets take head of bitcoin data and add it a time list
data2=data.head()
date_list =["2018-01-16","2018-02-19","2018-03-09","2018-04-17","2018-09-19"]
datetime_object=pd.to_datetime(date_list)
data2["date"]=datetime_object
# lets make date as index
data2=data2.set_index("date")
data2 


#now we can select according to our index
print(data2.loc["2018-02-19"])
print(data2.loc["2018-01-01":"2018-06-01"])
data2.resample("A").mean() #We will use data2 that we create at previous part
# Lets resample with month
data2.resample("M").mean() 

# In real life (data is real. Not created from us like data2) we can solve this problem with interpolate
# We can interpolete from first value
data2.resample("M").first().interpolate("linear")
# Or we can interpolate with mean()
data2.resample("M").mean().interpolate("linear")
 #data =pd.read_csv("../input/bitflyerJPY_1-min_data_2017-07-04_to_2018-06-27.csv"
#data=data.set_index("#")
#data.head()
#indexing using square brackets
data["Open"][1]
# using column attribute and row label
data.Open[1]
# using loc accessor
data.loc[1,["Open"]]
data[["Open","Close"]]

# Difference between selecting columns: series and dataframes
print(type(data["Open"])) #series
print(type(data[["Open"]])) #dataframes
# Slicing and indexing series
data.loc[1:10,"Open":"Close"]  #10 and "Close" are inclusive

# Reverse slicing 
data.loc[10:1:-1,"Open":"Close"] #
# From something to end
data.loc[1:10,"Low":]
#creating booelan series
boolen =data.Open >300000
data[boolen]
#combining filters
first_filter =data.Open<1500000
seconde_filte=data.Close>300000
data[first_filter & seconde_filte]
# Filtering column based others
data.Open[data.Low<3000000]
#plain function
def div(n):
    return n/2
data.Timestamp.apply(div)
# Or we can use lambda function
data.Timestamp.apply(lambda x : x/2)
# Defining column using other columns
data["total_difference"]= data.Close -data.Open
data.head()
# our index name is this:a
#data.index.name="#"
print(data.index.name)
# lets change it
data.index.name = "index_name"
data

# Overwrite index
# if we want to modify index we need to change all of them.
data.head()
# first copy of our data to data4 then change index 
data4 = data.copy()
# lets make index start from 100. It is not remarkable change but it is just example
#data4.index = range(100,150,3)
data4.head()

# We can make one of the column as index. I actually did it at the beginning of manipulating data 
# data= data.set_index("#")
# also you can use 
# data.index = data["#"]
#lets read dara frame one more time to start from beginning
data =pd.read_csv("../input/bitflyerJPY_1-min_data_2017-07-04_to_2018-06-27.csv")
data.head()
# Setting index : type 1 is outer type 2 is inner index
data1=data.set_index(["Open","Close"])
data1.head(100)
#data1.loc["Low","High"] # howw to use indexes
dic = {"treament":["A","A","B","B"],"gender":["F","M","F","M"],"response":[12,30,45,68],"age":[20,49,18,69]}
df = pd.DataFrame(dic)
df
#pivoting
df.pivot(index="treament",columns= "gender",values="age"),

df1=df.set_index(["treament","gender"])
df1
# lets unstack it
#level determines indexes
df1.unstack(level=0)

df1.unstack(level=1)
# change inner and outer level index position
df1.swaplevel(0,1)
df1
df
#df.pivot(index="treament",columns="gender",values="age")
pd.melt(df,id_vars="treament",value_vars=["age","response"])
#we will use df
df
# according to treatment take means of other features
df.groupby("treament").mean()   
# there are other methods like sum, std,max or min
# we can only choose one of the feature
df.groupby("treament").age.max()
# Or we can choose multiple features
df.groupby("treament")[["age","response"]].min()
df.info()
# as you can see gender is object
# However if we use groupby, we can convert it categorical data. 
# Because categorical data uses less memory, speed up operations like groupby
df["gender"] = df["gender"].astype("category")
df["treament"] = df["treament"].astype("category")
df["response"]= df["response"].astype("float")
df.info()
