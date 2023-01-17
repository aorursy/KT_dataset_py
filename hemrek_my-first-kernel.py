# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visulation

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/pokemon.csv") # read data
# info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage
data.info()
# columns gives column names of features
data.columns
# shape gives number of rows and columns in a tuble
data.shape
# head shows first 5 rows
data.head()
# tail shows last 5 rows
data.tail()
data.corr()
data.describe()
print(data['Type 2'].value_counts(dropna =False))
data["Type 1"].unique()
data["Type 1"] == "Dragon" # We can filter the data if we want 
data[data["Type 1"] == "Dragon"]
x = (data.Speed > 140) & (data.Attack >80) # We can use & or | 
data[x]
a = data.Legendary == True # filter the booleans("True","False")
data[a]
f,(ax1,ax2) = plt.subplots(2,1,figsize= (15,15),sharex=True)
data.Attack.plot(kind="line",color="red",label="Attack",grid=True,linewidth=1,ax=ax1)
ax1.set_ylabel("Attack")
ax1.legend()
ax1.set_title("Attack Graphic")

data["Sp. Atk"].plot(kind = "line",color="blue",label="Sp. Atk",grid=True,linestyle=":",ax=ax2)
ax2.set_title("Sp. Atk Graphic")
plt.xlabel("Pokemons")
plt.ylabel("Sp. Atk")
ax2.legend()
plt.show()
figure = plt.subplots(1,figsize=(15,15))

plt.subplot(211)
plt.bar(data["Type 1"],data.Speed,label="Speed",color="green",width=.7)
plt.xlabel("Pokemon's Type")
plt.ylabel("Pokemon's Speed")
plt.title("Speed Bars")
plt.legend()

plt.subplot(212)
plt.bar(data["Type 1"],data.HP,label="HP",color="red",width=.7)
plt.xlabel("Pokemon's Type")
plt.ylabel("Pokemon's HP")
plt.title("HP Bars")
plt.legend()
plt.show()

plt.subplots(figsize=(10,5))
plt.scatter(data["Sp. Atk"],data["Sp. Def"],color=("blue","red"))
plt.xlabel("Sp. Atk")
plt.ylabel("Sp. Def")
plt.legend()
plt.show()
plt.subplots(figsize=(15,12))
plt.subplot(211)
plt.hist(data.Speed,color="orange",bins=20,histtype="bar",orientation="horizontal") # Horizontal Histogram
plt.xlabel("Frequency")
plt.ylabel("Speed")
plt.title("Horizontal Histogram")

plt.subplot(212)
plt.hist(data.Speed,color="blue",bins=20,histtype="bar",orientation="vertical") # vertical Histogram
plt.xlabel("Speed")
plt.ylabel("Frequency")
plt.title("Vertical Histogram")
plt.show()

# For example: compare attack of pokemons that their Type
# Black line at top is max
# Blue line at top is 75%
# Red line is median (50%)
# Blue line at bottom is 25%
# Black line at bottom is min
# black circles are outliers
f,ax=plt.subplots(1,1,figsize=(15,8))
data.boxplot(column="Defense",by="Type 1",ax=ax)
plt.show()
mean_attack = sum(data.Attack)/len(data.Attack)
mean_attack
data["sum_attack"]=["high" if i>mean_attack else "low" for i in data.Attack]
data.loc[:10,("sum_attack","Attack")]
mean_defense = sum(data.Defense)/len(data.Defense)
mean_defense
data["sum_defense"]=["high" if i>mean_defense else "low" for i in data.Defense]
data.loc[:10,("sum_defense","Defense")]
def summation(x,y,z):
    """return summation of 3 numbers"""
    total =x+y+z
    return total
     
summation(1,5,7)
import random

def summation():
    def rand():
        x = random.randint(1,101)
        y = random.randint(1,101)
        z = random.randint(1,101)
        print("random numbers:",x,y,z)
        return x,y,z
    summ = 0
    for i in rand():
        summ = summ +i
    return summ

print("")
print(summation())

# default arguments
def summation(x,y,z=5):# z is a default argument but you can define it
    return x+y+z

print(summation(2,7))
print("")
print(summation(2,7,2))
# flexible arguments *args
def f(*args):
    """print number in args"""
    for i in args:
        print(i)
        
f(7)
print("")
f(2,7,5,6,5)

# flexible arguments **kwargs that is dictionary
def f(**kwargs):
    """ print key and value of dictionary"""
    for key, value in kwargs.items():
        print(key, " ", value)
f(name1 = 'matthew', age1 = 26, name2 = 'charles', age2 = 14)
summation = lambda x,y,z: x+y+z
print(summation(1,7,8))

square = lambda x: x*x
print(square(7))
number_list = [5,8,9,2,3]
y = map(lambda x: x+5, number_list)
print(list(y))
data1 = data.head()
data1
melted = pd.melt(data1,id_vars=["Name"],value_vars=["Attack","Defense"])
melted
melted.pivot(index="Name",columns="variable",values="value")
data2 = data.head()
data3 = data.tail()

concDataV = pd.concat([data2,data3],axis=0,ignore_index=True) # vertical
concDataV
df1 = data.HP.head(10)
df2 = data.Speed.head(10)
df3 = data.Attack.head(10)
conc_data_h = pd.concat([df1,df2,df3],axis=1) # horizontal
conc_data_h
# data frames from dictionary
employee = ["Charles","Matthew","Mike"]
salary = [100,150,150]
list_label = ["employee","salary"]
list_col = [employee,salary]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped) 
df = pd.DataFrame(data_dict,index = [1,2,3]) # we can set index
df
# add new columns
df["workingHours"] = [5,7,6]
df
# Broadcasting
df["Test"] = 0   #Broadcasting entire column
df
time_list = ["1992-03-08","1992-04-12"]
print(type(time_list[1])) # As you can see date is string
# however we want it to be datetime object
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))
# In order to practice lets take head of pokemon data and add it a time list
data2 = data.head().copy()
date_list = ["2002-06-15","2002-06-21","2002-07-15","2003-08-12","2003-09-15"] # we create our time list
date_timeobject = pd.to_datetime(date_list)

data2["date"] = date_timeobject # lets set date as index
data2 = data2.set_index("date")
data2
print(data2.loc["2002-06-15"])
print("-"*80)
print(data2.loc["2002-06-15":"2002-07-15"])
# Lets resample with year
data2.resample("A").mean() # we can use other functions as min(),sum()
# Lets resample with month
data2.resample("M").mean()
# As you can see there are a lot of nan because data2 does not include all months
# We can interpolete from first value
data2.resample("M").first().interpolate("linear")
# Or we can interpolate with mean()
data2.resample("M").mean().interpolate("linear")
