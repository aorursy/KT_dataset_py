# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/pokemon-challenge/pokemon.csv")
data.info()
data.corr()
#correlation map



f,ax = plt.subplots(figsize=(12,12))

sns.heatmap(data.corr(),annot=True,linewidths=0.5,fmt='.1f',ax=ax)

plt.show()
data.head(10)
data.columns
#line

data.Speed.plot(kind="line",color="g",label="Speed",linewidth=1,alpha=0.5,grid=True,linestyle=":")

data.Defense.plot(color="r",label="Defense",linewidth=1,alpha=0.5,grid=True,linestyle="-.")

plt.legend(loc="upper right")

plt.xlabel("x axis")

plt.ylabel("y axis")

plt.title("Line Plot")
#scatter

data.plot(kind="scatter",x="Attack",y="Defense",alpha=0.5,color="red")

plt.xlabel("Attack")

plt.ylabel("Defense")

plt.title("Attack Defense Scatter Plot")
#scatter with another coding style

plt.scatter(data.Attack,data.Defense,color="red",alpha=0.5)

plt.xlabel("Attack")

plt.ylabel("Defense")

plt.title("Attack Defense Scatter Plot")
#histogram



data.Speed.plot(kind="hist",bins=50,figsize=(10,10))

plt.show()



#bin = barların sayısıdır.
# clf() = cleans it up again you can start a fresh



data.Speed.plot(kind="hist",bins=50,figsize=(15,15))

plt.clf()
#dictionaries are faster than lists
dictionary = {'spain':'madrid','usa':'vegas'}

print(dictionary.keys())

print(dictionary.values())
dictionary['spain'] = "barcelona"

print(dictionary)

dictionary['france'] = "paris"

print(dictionary)



del dictionary['spain']

print(dictionary)



print('france' in dictionary)

dictionary.clear()

print(dictionary)
del dictionary

print(dictionary)
data = pd.read_csv("/kaggle/input/pokemon-challenge/pokemon.csv")
series = data.Defense

print(type(series))

data_frame = data[['Defense']]

print(type(data_frame))
data[data.Defense > 200]
#filtering pandas with logical and



data[np.logical_and(data.Defense>200,data.Attack>100)] 

#filtering pandas with another code style



data[(data.Defense>200) & (data.Attack>100)]
i = 0

while i !=5:

    print("i is : ",i)

    i+=1

print(i," is equal to 5")
lis = [1,2,3,4,5]



for i in lis:

    print("i is : ",i)

print("")



#enumerate index and value of list



for index,value in enumerate(lis):

    print(index," : ",value)

print("")





#for dictionaries



dictionary = {'spain':'madrid','france':'paris'}



for key,value in dictionary.items():

    print(key," : ",value)

print("")



#for pandas we can achieve index and value



for index,value in data[["Attack"]][0:1].iterrows():

    print(index, " : ",value)
def tuble_ex():

    """ return defined t tuble"""

    t= (1,2,3)

    return t



a,b,c = tuble_ex()

print(a,b,c)
#global , local , ....
def square():

    

    def add():

        """add two local variable"""

        x=2

        y=3

        z=x+y

        return z

    return add()**2

print(square())
#default arguments example

def f(a,b=1,c=2):

    y = a+b+c

    return y

print(f(5))



print(f(5,4,3))
#iteration example



name="ronaldo"

it=iter(name)

print(next(it))

print(*it)
data = pd.read_csv("/kaggle/input/pokemon-challenge/pokemon.csv")

data.head()
data.tail()
data.columns
data.shape
data.info()
print(data["Type 1"].value_counts(dropna=False))
data.describe()
data.boxplot(column="Attack",by="Legendary")

plt.show()
data_new = data.head()

data_new
#id_vars = what we do not want to melt

#value_vars = what we want to melt







melted = pd.melt(frame=data_new,id_vars="Name",value_vars=["Attack","Defense"])

melted
#reversal of melting



melted.pivot(index="Name",columns="variable",values="value")
data1 = data.head()

data2 = data.tail()



conc_data_row = pd.concat([data1,data2],axis=0,ignore_index=True)

conc_data_row
data1 = data.Attack.head()

data2 = data.Defense.head()



conc_data_cols = pd.concat([data1,data2],axis=1)

conc_data_cols
data.dtypes
data["Type 1"] = data["Type 1"].astype("category")

data["Speed"] = data["Speed"].astype("float")
data.dtypes
data.head()



#for example charmender has type1 but no type2. thats means missing data. 

#may be ıt has bu didnt written.!!!
data.info()
data["Type 2"].value_counts(dropna=False)
data1=data

data1["Type 2"].dropna(inplace=True)
data["Type 2"].value_counts(dropna=False)
assert 1==1 #doğruysa birşey döndürmüyor!!
assert 1==2
assert data["Type 2"].notnull().all() #returns nothing because we drop nan values
data["Type 2"].fillna("empty",inplace=True)
assert data["Type 2"].notnull().all() # returns nothing because we do not have nan values
assert data.columns[1] == "Name"
assert data.Speed.dtypes == np.float
data_dict = {"Country":["Spain","France"],"Population":[11,12]}

df = pd.DataFrame(data_dict)

df
df["Capital"] = ["madrid","paris"]

df
df["Income"] = 0

df
data1 = data.loc[:,["Attack","Defense","Speed"]]

data1.plot()
data1.plot(subplots=True)
data1.plot(kind="scatter",x="Attack",y="Defense")
data1.plot(kind="hist",y="Defense",bins=50,range=(0,250),normed=False)



# normed = normalize the data btwn 0-1
# histogram subplot with non cumulative and cumulative

fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[0])

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)

time_list = ["1992-03-08","1992-04-12"]

print(type(time_list[1])) # As you can see date is string

# however we want it to be datetime object

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
# close warning

import warnings

warnings.filterwarnings("ignore")

# In order to practice lets take head of pokemon data and add it a time list

data2 = data.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object

# lets make date as index

data2= data2.set_index("date")

data2 
print(data2.loc["1993-03-16"])

print(data2.loc["1992-03-10":"1993-03-16"])
# We will use data2 that we create at previous part

data2.resample("A").mean()
# Lets resample with month

data2.resample("M").mean()

# As you can see there are a lot of nan because data2 does not include all months
data.head()
# read data

data = pd.read_csv("/kaggle/input/pokemon-challenge/pokemon.csv")

data= data.set_index("#")

data.head()
data["HP"][1]
data.loc[1,["HP"]]
data[["HP","Attack"]]
# Difference between selecting columns: series and dataframes

print(type(data["HP"]))     # series

print(type(data[["HP"]]))   # data frames
# Slicing and indexing series

data.loc[1:10,"HP":"Defense"]   # 10 and "Defense" are inclusive
# Reverse slicing 

data.loc[10:1:-1,"HP":"Defense"] 
# From something to end

data.loc[1:10,"Speed":] 
# Creating boolean series

boolean = data.HP > 200

data[boolean]
# Combining filters

first_filter = data.HP > 150

second_filter = data.Speed > 35

data[first_filter & second_filter]
data[data.Speed<15]
# Filtering column based others

data.HP[data.Speed<15]
# Plain python functions

def div(n):

    return n/2

data.HP.apply(div)
# Or we can use lambda function

data.HP.apply(lambda n : n/2)
# Defining column using other columns

data["total_power"] = data.Attack + data.Defense

data.head()
# our index name is this:

print(data.index.name)

# lets change it

data.index.name = "index_name"

data.head()
# Overwrite index

# if we want to modify index we need to change all of them.

data.head()

# first copy of our data to data3 then change index 

data3 = data.copy()

# lets make index start from 100. It is not remarkable change but it is just example

data3.index = range(100,900,1)

data3.head()
# lets read data frame one more time to start from beginning

data = pd.read_csv("/kaggle/input/pokemon-challenge/pokemon.csv")

data.head()

# As you can see there is index. However we want to set one or more column to be index
# Setting index : type 1 is outer type 2 is inner index

data1 = data.set_index(["Type 1","Type 2"]) 

data1.head(50)

# data1.loc["Fire","Flying"] # howw to use indexes
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df = pd.DataFrame(dic)

df
# pivoting

df.pivot(index="treatment",columns = "gender",values="response")
df1 = df.set_index(["treatment","gender"])

df1

# lets unstack it
# level determines indexes

df1.unstack(level=0)
df1.unstack(level=1)