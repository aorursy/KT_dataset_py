import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data= pd.read_csv("../input/anime.csv")
data.info()

x=2

def f():

    x= np.mean(data.rating)

    return x

print(x)

print(f())
# if there is no local scope,it uses the global one!!

x=2

def f():

    y= x* (np.mean(data.rating))

    return y

print(f())
# built in scope

import builtins

dir(builtins)
#nested function

def f1():

    

    def f2():

        x= np.max(data.rating)

        y= data.rating.mean()

        z= x+y

        return z

    return f2()**2

  

print(f1())
# default arguments

def multiply(a, b, c= np.mean(data.members)):

    x= a * b * c

    return x



print(multiply(2,3))

print(multiply(2,3,4))
# lambda function  ---->  faster way of writing function

multiply= lambda a,b,c= np.mean(data.members): a*b*c

print(multiply(2,3))

print(multiply(2,3,4))

#anonymous functions ----> similiar to lambda function but this can take more than one arguments.

#ex : map(func,seq)  ----> applies a function to all the items in a list

number_list=[2,3,4]

x= map( lambda a: a*np.min(data.members) ,number_list)

print(list(x))
# flexible arguments *args

def f(*args):

    for i in args:

        print(i*i)

f(1)

f(1,2)

f(1,2,3)

print("------")

# flexible arguments **kwargs that is dictionary

def f(**kwargs):

    for key, value in kwargs.items():  

        print(key, ":", value)

        

f(name = 'ali', surname = 'veli', id = '1234567890')
# iteration 

name = "Anime"

it = iter(name)

print(next(it))

print(*it)        

#zip(): zip lists

list1 = [1,2,3,4]

list2 = [5,6,7,8]

z = zip(list1,list2)

print(z)

zip_list = list(z)

print(zip_list)
# Unzip

un_zip = zip(*zip_list)

un_list1,un_list2 = list(un_zip) # be careful,it returns tuple!!

print(un_list1)

print(un_list2)



print(type(un_list2)) #control.

print(list(un_list1))

print(list(un_list1))

#list compherension

a = [1,2,3]

b = [i + 1 for i in a]

print(b)
#list compherension

num1 = data.anime_id.tolist()

num2 = [i/5 for i in num1 ]

print(num2)
x=9.17

data["popularity"]=["Very Good"if i>x else "Good"if i==x else "Ok" for i in data.rating]

data.loc[:5,["rating","popularity"]]

data.head()
data.columns
data.shape
#frequency of anime types.

print(data['type'].value_counts(dropna=False))
data.describe()
data.boxplot(column='anime_id',by='popularity') # by--->groupby()

plt.show() 
data_new= data.head()

data_new
# melt

melted = pd.melt(frame=data_new,id_vars = 'name', value_vars= ['anime_id','rating'])

melted

#pivoting

#reverse of melting

melted.pivot(index = 'name', columns = 'variable',values='value')
data1 = data.head()

data2= data.tail()

conc_data= pd.concat([data1,data2],axis =0,ignore_index =True) #ignor_index--->1,2,3,4

conc_data
data1 = data['genre'].head()

data2= data['type'].head()

conc_data_col = pd.concat([data1,data2],axis =1) 

conc_data_col
data.dtypes
data['members'] = data['members'].astype('float')

data.dtypes
data["type"].value_counts(dropna =False)

#droping nana values

data1=data

data1["type"].dropna(inplace = True)
#control!!

assert 1==1
assert data['type'].notnull().all()
data['type'].fillna('empty',inplace=True)
assert  data['type'].notnull().all()
assert data.columns[0] == 'anime_id' #true
assert data.columns[1] == 'anime_id' #false
assert data.name.dtypes == np.object
# data frames from dictionary

name=["Kimi no Na wa","Gintama"]

rating=["9.17","8,15"]

list_label=["name","rating"]

list_col=[name,rating]

ziped=list(zip(list_label,list_col))

data_dict=dict(ziped)

df=pd.DataFrame(data_dict)

df
# Add new columns

df["type"]=["Movie","Tv"]

df
df["members"]= 5

df
# Visual exploratory

data= pd.read_csv("../input/anime.csv")
data_new= data.loc[:,["anime_id","members"]]

data_new.plot()

plt.show()
data_new.plot(subplots=True)

plt.show()
data_new.plot(kind="scatter",x="anime_id",y="members")

plt.show()
data_new.plot(kind="hist",y="members",bins=50, range=(0,250), density=True,color="orange")

plt.show()
# histogram subplot with non cumulative and cumulative

fig,axes= plt.subplots(nrows=2,ncols=1)

data_new.plot(kind="hist",y="members",bins=50, range=(0,250), density=True,color="orange",ax=axes[0])

data_new.plot(kind="hist",y="members",bins=50, range=(0,250), density=True,color="orange",ax=axes[1],cumulative=True)

plt.savefig('graph.png')

plt

plt.show()

time_list = ["1992-03-08","1992-04-12"]

print(type(time_list[1]))

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object)) #str to datetime object
# close warning

import warnings

warnings.filterwarnings("ignore")



data2 = data.head()

date_list = ["1996-12-11","1996-11-12","1996-10-13","1996-08-14","1996-09-15"]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object



data2= data2.set_index("date")

data2 
print(data2.loc["1996-10-13"])

print("*************************")

print(data2.loc["1996-08-14":"1996-09-15"])
data2.resample("A").mean() # A= "year" M="month"
data2.resample("M").mean()
data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")
#indexing

#read the data

data = pd.read_csv('../input/anime.csv')

print(data.index.name)
data.index.name="#"

data.head()
data["anime_id"][1]
data.anime_id[1]
data[["anime_id","members"]]
#slicing data frame

print(type(data["rating"]))     # series

print(type(data[["rating"]]))   # data frames
# Slicing and indexing series

data.loc[1:6,"anime_id":"rating"]  
# Reverse slicing 

data.loc[7:1:-1,"anime_id":"rating"] 
# From something to end

data.loc[1:6,"genre":] 
# Creating boolean series

boolean= data.rating > 9.11

data[boolean]
# Combining filters

b1= data.rating > 9.11

b2= data.anime_id < 15000

data [b1 & b2]

# Filtering column based others

data.anime_id[data.rating>9.2]
def div(n):

    return n/2

data.rating.apply(div)
data.rating.apply(lambda x:x/2)
# Defining column using other columns

data["total"] = data.anime_id + data.rating

data.head()
data3 = data.copy()

data3.index = range(100,12394,1)

data3.head()
data.head()
# Setting index

data1 = data.set_index(["rating","type"]) 

data1.head(20)

#pivoting -----> reshape tool

dic= {"name":["Kimi no Na wa","Gintama","Mushishi"],"rating":["9.17","8,15","6.6"],"episodes":["1", "51","26"]}

df=pd.DataFrame(dic)

df
df.pivot(index="name",columns="episodes",values="rating")
df1 =df.set_index(["name","rating"])

df1
#unstack 

df1.unstack(level=0)
df1.unstack(level=1)
df2 = df1.swaplevel(0,1)

df2
df
# reverse pivoting

#df.pivot(index="name",columns="episodes",values="rating")

pd.melt(df,id_vars="name",value_vars=["episodes","rating"])
data.groupby("type").mean()
data.groupby("type").rating.max()
data.groupby("type")[["rating","members"]].min() 