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
data = pd.read_csv("../input/pokemon.csv")
data.info()
data.describe()
f,ax = plt.subplots(figsize=(15,15))

sns.heatmap(data.describe(), annot=True, linewidths=3, fmt=".2f", ax=ax)

plt.show()



# sns.heatmap(data.describe())
data.corr()
f,ax = plt.subplots(figsize=(15,15)) # Boyutu belirliyoruz.

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax )

plt.show()



#  f -> figure to be created



#  ax -> a matplotlib.axes.Axes instance to which the heatmap is plotted. 

# If not provided, use current axes or create a new one.
data.head(10)

#data.loc[:10,:]
data.columns
# Line plot



data.Speed.plot(kind='line', color="green", label="Speed", linewidth=1, alpha=0.5, grid=True, linestyle=':')

data.Defense.plot(kind='line', color='red', label="Defense", linewidth=1, alpha=0.5, grid=True, linestyle="-.")

plt.legend(loc='upper right')

plt.xlabel("X axis")

plt.ylabel("Y axis")

plt.title("Line Plot")

plt.show()



plt.subplot(1,2,1)

plt.plot(data["Speed"], color="green",linewidth=0.6)

plt.title("Speed")



plt.subplot(1,2,2)

plt.plot(data["Defense"], color="red",linewidth=0.6)

plt.title("Defense")



plt.show()
dta = pd.DataFrame()



dta["Efsanevi"]=[1 if each==True else 0 for each in data.Legendary] 

# dta["Efsanevi"] = data.Legendary.astype(int)



dta.Efsanevi.hist()

#plt.hist(dta["Efsanevi"],bins=3)

plt.title("Legendary")

plt.show()
count = 0

for each in data.Legendary:

    if each==True:

        count = count+1

    else:

        continue

print("number of legendary pokemon=",count)



dta.Efsanevi.value_counts()
data["Type 1"].unique()
type_sum = data["Type 1"].value_counts()



x = type_sum.size



type_sum.hist(bins=x)
type_sum
liste = []



for each in data["Type 1"]:

    if each=="Grass":

        liste.append(0)

    elif each=="Fire":

        liste.append(1)

    elif each=="Water":

        liste.append(2)

    elif each=="Bug":

        liste.append(3)

    elif each=="Normal":

        liste.append(4)

    elif each=="Poison":

        liste.append(5)

    elif each=="Electric":

        liste.append(6)

    elif each=="Ground":

        liste.append(7)

    elif each=="Fairy":

        liste.append(8)

    elif each=="Fighting":

        liste.append(9)

    elif each=="Psychic":

        liste.append(10)

    elif each=="Rock":

        liste.append(11)

    elif each=="Ghost":

        liste.append(12)

    elif each=="Ice":

        liste.append(13)

    elif each=="Dragon":

        liste.append(14)

    elif each=="Dark":

        liste.append(15)

    elif each=="Steel":

        liste.append(16)

    else: # Flying

        liste.append(17)



dta["Turler_Type1"] = liste

#plt.hist(dta["Türler_Type2"],bins=38)

dta["Turler_Type1"].hist(bins=18)

plt.show()



del(liste)
data["Type 2"].unique()
liste = []



for each in data["Type 2"]:

    if each=="Grass":

        liste.append(0)

    elif each=="Fire":

        liste.append(1)

    elif each=="Water":

        liste.append(2)

    elif each=="Bug":

        liste.append(3)

    elif each=="Normal":

        liste.append(4)

    elif each=="Poison":

        liste.append(5)

    elif each=="Electric":

        liste.append(6)

    elif each=="Ground":

        liste.append(7)

    elif each=="Fairy":

        liste.append(8)

    elif each=="Fighting":

        liste.append(9)

    elif each=="Psychic":

        liste.append(10)

    elif each=="Rock":

        liste.append(11)

    elif each=="Ghost":

        liste.append(12)

    elif each=="Ice":

        liste.append(13)

    elif each=="Dragon":

        liste.append(14)

    elif each=="Dark":

        liste.append(15)

    elif each=="Steel":

        liste.append(16)

    elif each=="Flying":

        liste.append(17)

    else: # Normal

        liste.append(18)



dta["Turler_Type2"] = liste

#plt.hist(dta["Türler_Type2"],bins=38)

dta["Turler_Type2"].hist(bins=19)

plt.show()



del(liste)
dta.columns
dta['Turler_Type1'].plot(kind="hist", alpha=.8)

dta['Turler_Type2'].plot(kind="hist", alpha=0.5, color="yellow")



# blue + yellow -----> green :)
count = 0

for each in data["Type 1"]:

    if each=="Water":

        count = count+1

    else:

        continue

print("number of water pokemon=",count,"\n")



x = data["Type 1"].value_counts()

print("number of water pokemon=",x["Water"])

print("number of fire pokemon=",x["Fire"])
filtre1 = data["Type 1"] == "Water"

mean_point = data[filtre1].mean()

mean_point
x = data["Type 1"] == "Water"

# x

data[x]
print("Max attack = ",max(data["Attack"]))



print("Max attack fire pokemon = ",max(data.Attack[data["Type 1"] == "Fire"]))

print("Max attack legendary pokemon = ",max(data.Attack[data.Legendary == True]))

print("Max attack 5th generation pokemon = ",max(data.Attack[data.Generation == 5]))
grass_pokemons = data["Type 1"] == "Grass"

generation_5_pokemons = data.Generation == 5



x = data[generation_5_pokemons & grass_pokemons]

print("max attack genertion 5 grass pokemon =",max(x.Attack))
data_water = data[data["Type 1"]=="Water"]

data_water.head()
water_hp_mean = data_water.HP.mean()



water_attack_mean = data_water.Attack.mean()



water_defense_mean = data_water.Defense.mean()



water_speed_mean = data_water.Speed.mean()



water_spatk_mean = data_water["Sp. Atk"].mean()



water_spdef_mean = data_water["Sp. Def"].mean()



print("Water pokemons above the general average")

data_water[(data_water.HP >= water_hp_mean) 

            & (data_water.Attack >= water_attack_mean) 

            & (data_water.Defense >= water_defense_mean) 

            & (data_water.Speed >= water_speed_mean)

            & (data_water["Sp. Atk"] >= water_spatk_mean)

            & (data_water["Sp. Def"] >= water_spdef_mean)]
hp_mean = data.HP.mean()



attack_mean = data.Attack.mean()



defense_mean = data.Defense.mean()



speed_mean = data.Speed.mean()



spatk_mean = data["Sp. Atk"].mean()



spdef_mean = data["Sp. Def"].mean()



print("Pokemons above the general average")

data_water[(data.HP >= hp_mean) 

            & (data.Attack >= attack_mean) 

            & (data.Defense >= defense_mean) 

            & (data.Speed >= speed_mean)

            & (data["Sp. Atk"] >= spatk_mean)

            & (data["Sp. Def"] >= spdef_mean)]
data.plot(kind="scatter", x="Attack", y="Defense", alpha=0.5, color="purple",figsize=(7,7))

plt.xlabel("Attack")

plt.ylabel("Defense")

plt.title("Attack - Defense Scatter Plot")

plt.show()
grass = data["Type 1"] == "Grass"

x = data.Attack[grass]

y = data.Defense[grass]

plt.scatter(x,y, color="green", alpha=.5)



water = data["Type 1"] == "Water"

x = data.Attack[water]

y = data.Defense[water]

plt.scatter(x,y, color="blue", alpha=.5)



fire = data["Type 1"] == "Fire"

x = data.Attack[fire]

y = data.Defense[fire]

plt.scatter(x,y, color="red", alpha=.5)



plt.xlabel("Attack")

plt.ylabel("Defense")

plt.title("Water, Fire, Grass Pokemons Attack and Defense")

plt.show()
water = data["Type 1"] == "Water"

y = data.Speed[water]

x = data["#"][water]

plt.scatter(x,y, color="blue", alpha=.7)



fire = data["Type 1"] == "Fire"

y = data.Speed[fire]

x = data["#"][fire]

plt.scatter(x,y, color="red", alpha=.7)



grass = data["Type 1"] == "Grass"

y = data.Speed[grass]

x = data["#"][grass]

plt.scatter(x,y, color="green", alpha=.7)



data.Speed.plot(kind="line", color="purple", alpha=0.3, figsize=(13,8))

plt.title("Speed of all pokemons  /  Water,Fire,Grass pokemons speed")

plt.xlabel("Pokemon id")

plt.ylabel("Speed")

plt.show()
data["Type 1"].unique()
#data.Speed.plot(kind="hist",bins=50, figsize=(10,10))



data["Speed"].hist(bins=50,figsize=(10,10))
data.columns
data.Attack.plot(kind="hist", alpha=.8,grid=True)

data.Defense.plot(kind="hist", alpha=0.5, color="yellow",grid=True)

plt.legend()

plt.title("Attack - Defense")



# blue + yellow -----> green :)
# clf()



data.Speed.plot(kind="hist",bins=50, figsize=(10,10))

plt.clf()
data.plot(kind='scatter', x='#', y='Defense', alpha=.8, color='red', figsize=(15,10))

data.Defense.plot(color='blue', label='Defence', linewidth=1, alpha=.5, grid=True, linestyle='-.')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
# DICTIONARY



dictionary = {'spain' : 'madrid','usa' : 'vegas'}

print(dictionary.keys())

print(dictionary.values())
dictionary["spain"] = "barcelona"

print(dictionary)



dictionary["france"] = "paris"

print("dictionary")



del dictionary["spain"]



print(dictionary)

print("france" in dictionary)



dictionary.clear()

print(dictionary)
# del dictionary    # delete entire dictionary     

print(dictionary)   # it gives error because dict
# PANDAS



data = pd.read_csv("../input/pokemon.csv")
series = data["Defense"]

print(type(series))



data_frame = data[["Defense"]]

print(type(data_frame))
# Comparison operator

print(3 > 2)

print(3!=2)

# Boolean operators

print(True and False)

print(True or False)
x = data["Defense"] > 200



data[x]
data[np.logical_and(data["Defense"]>200, data["Attack"]>100)]
data[(data["Defense"]>200) & (data["Attack"]>100)]
# WHILE and FOR LOOPS



i = 0

while i != 5 :

    print('i is: ',i)

    i +=1 

print(i,' is equal to 5')
lis = [1,2,3,4,5]

for i in lis:

    print("i is: ",i)

print("")



# Enumerate

for index,value in enumerate(lis):

    print(index," : ",value)

print("")



dic = {"spain":"madrid", "turkey":"istnbul","ABD":"New York"}

for key,value in dic.items():

    print(key," : ",value)

print("")



for index,value in data[["Attack"]][0:1].iterrows():

    print(index," : ",value)
# example of what we learn above

def tuble_ex():

    """ return defined t tuble"""

    t = (1,2,3)

    return t

a,b,c = tuble_ex()

print(a,b,c)



g = tuble_ex()

print(g)
# SCOPE

# global ,local, built in scope



x=2

def f():

    x=3

    return x

print(x)

print(f())
x=5

def f():

    y = 2*x

    return y



print(f())
import builtins

dir(builtins)



# built in scope = len,max,round
# NESTED FUNCTION



def square():

    """return square of vlaue"""

    def add():

        """add two local variable"""

        x = 2

        y = 3

        z = x+y

        return z

    return add()**2

print(square())
# DEFAULT and FLEXIBLE ARGUMENTS



# DEFAULT

print("Default")

def f(a,b = 3,c=2):

    y = a+b+c

    return y

print(f(5))

print(f(5,20,100))



# FLEXIBLE

print("\nFLEXIBLE")

def f(*args):

    for i in args:

        print(i)

f(1)

print("")

f(4,6,3,77,9)
def f(**kwargs):

    """ print key and value of dictionary"""

    for key,value in kwargs.items():

        print(key," : ",value)

f(country = 'spain', capital = 'madrid', population = 123456)
# Lambda



square = lambda x:x**2

print(square(7))



tot = lambda x,y,z:x+y+z

print(tot(8,9,123))
# ANONYMOUS FUNCTİON



number_liste = [1,2,3]

y = map(lambda x:x**2,number_liste)

print(list(y))
# ITERATORS



# iteration example

name = "ronaldo"

it = iter(name)

print(next(it))

print(next(it))

print(*it)
# zip()



liste1 = [1,2,3,4,5]

liste2 = [6,7,8,9,10]



z = zip(liste1,liste2)

z_list = list(z)

print(z_list)
# un_zip



un_zip = zip(*z_list)

un_liste1,un_liste2 = list(un_zip)

print(un_liste1)

print(un_liste2)

print(type(un_liste2))



print(type(list(un_liste2))) # tuple ---> list
num1 = [1,2,3]

num2 = [i + 1 for i in num1 ]

print(num2)
num1 = [1,2,3,4,5,6,7,8,9,10,11,12]

num2 = [i**2 if i == 10 else i-5 if i < 7 else i+5 for i in num1]

print(num2)
# speed ortalamasını al

ort = sum(data["Speed"]) / len(data["Speed"])

print(ort)



data["Speed_Level"] = ["high" if each>ort else "low" for each in data.Speed]

data.loc[:10,["Speed","Speed_Level"]]
data = pd.read_csv("../input/pokemon.csv")

data.head()
#data.Type 1.head() #error

data["Type 1"].head()



#data.tail()

#data.columns

#data.shape

#data.info()

#data.corr()

#data.describe()
print(data["Type 1"].value_counts(dropna=False))
#plt.hist(data["Type 1"].value_counts(dropna=False),bins=18)

#plt.show()



x = data["Type 1"].value_counts(dropna=False)

x.hist(bins=18)
data.describe()
data.boxplot(column="Attack", by="Legendary")

plt.show()



data.boxplot(column="Attack",by="Type 1", figsize=(15,10))

plt.show()
data_new = data.head()

data_new
melted = pd.melt(frame=data_new, id_vars="Name", value_vars=["Attack","Defense","Speed"]) 

melted
melted.pivot(index="Name", columns="variable",values="value")
dt1 = data.head()

dt2 = data.tail()

conc_data_row = pd.concat([dt1,dt2],axis=0,ignore_index=True)

conc_data_row
dt1 = data["Attack"].head()

dt2 = data["Defense"].head()

conc_data_col = pd.concat([dt1,dt2],axis=1)

conc_data_col
data.dtypes
# veri tiplerini değiştirelim...

data["Type 1"] = data["Type 1"].astype("category")

data["Speed"] = data["Speed"].astype("float")



data.dtypes
x = data["Name"] == "Charmander"

data[x]



# Charmander [Type 2] = NaN ??
data.info()



# Type 2 -----> 414 non-null object
data["Type 2"].value_counts(dropna=False)



# Type 2 -----> 386 non-null object
dt1 = data

dt1["Type 2"].dropna(inplace=True)
assert 1==1 # bir şey döndürmezse sorun yok demek...
# assert 1==2 # hata dönücek...
assert data["Type 2"].notnull().all() # hata yok
data["Type 2"].fillna("empty",inplace = True)
assert dt1["Type 2"].notnull().all()
# Assert ifadesiyle birçok şeyi kontrol edebiliriz. Örneğin

# assert data.columns[1] == 'Name'

# assert data.Speed.dtypes == np.int
# data frames from dictionary



country = ["Spain","France"]

population = ["11","21"]

list_lbel = ["country","popultion"]

list_col = [country,population]

zipped = list(zip(list_lbel,list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df



# list ----> dictionary ----> data frame
df["captial"] = ["Madrid","Paris"]

df
df["income"] = 0

df
data1 = data.loc[:,["Attack","Defense","Speed"]]

data1.plot(alpha = 0.8)
data1.plot(subplots = True)

plt.show()
data1.plot(kind="scatter", x="Attack", y="Defense")

plt.show()
data1.plot(kind="hist", y = "Defense", bins=50, range=(0,250),normed=True)

#plt.show()
# histogram subplot with non cumulative and cumulative

fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[0])

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt
data.describe()
time_list = ["1992-03-04","2010-11-09"]

print(type(time_list[1])) # str



datetime_object = pd.to_datetime(time_list)

print(type(datetime_object)) # DatetimeIndex
# close warning

import warnings

warnings.filterwarnings("ignore")



# In order to practice lets take head of pokemon data and add it a time list

data2 = data.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object



# lets make date as index

data2 = data2.set_index("date")

data2
# Now we can select according to our date index

print(data2.loc["1993-03-16"],"\n")



print("************************************\n")



print(data2.loc["1992-03-10":"1993-03-16"])
data2.resample("A").mean()
data2.resample("M").mean()
# In real life (data is real. Not created from us like data2) we can solve this problem with interpolate

# We can interpolete from first value

data2.resample("M").first().interpolate("linear")
# Or we can interpolate with mean()

data2.resample("M").mean().interpolate("linear")
# read data

data = pd.read_csv("../input/pokemon.csv")

data = data.set_index("#") # index 1 den başlıyacak

data.head()
data["HP"][1]

#data.HP[1]
data.loc[1,["HP"]]
data[["HP","Attack"]]
# Slicing data frame

print(type(data["HP"]))

print(type(data[["HP"]]))
data.loc[1:10,"HP":"Defense"]
# Reverse slicing 

data.loc[10:1:-1,"HP":"Defense"] 
data.loc[1:10,"Speed"]
# FILTERING DATA FRAMES



boolean = data.HP > 200

data[boolean]
first = data.HP > 150

second = data.Speed > 35

data[first & second]
# Filtering column based others

data.HP[data.Speed<15]



# hızı 15'ten küçük olanların canlarını ver
# TRANSFORMING DATA



print(data.loc[1:3,"HP"])



def div(n):

    return n/2

data.HP.apply(div)
data.HP.apply(lambda n:n/2)
data["totalPower"] = data.Attack + data["Sp. Atk"]

data.head()
print(data.index.name)



data.index.name = "index_name"

data.head()
data3 = data.copy()



data3.index = range(100,900,1)

data3.head()
# We can make one of the column as index. I actually did it at the beginning of manipulating data frames with pandas section

# It was like this

# data= data.set_index("#")

# also you can use 

# data.index = data["#"]
data = pd.read_csv("../input/pokemon.csv")

data.head()
data1 = data.set_index(["Type 1","Type 2"])

data1.head(100)
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df = pd.DataFrame(dic)

df
# pivot



df.pivot(index="treatment",columns="gender",values="response")
df1 = df.set_index(["treatment","gender"])

df1
df1.unstack(level=0)
df1.unstack(level=1)
df2 = df1.swaplevel(0,1)

df2
# MELTING DATA FRAMES



df
pd.melt(df,id_vars="treatment",value_vars=["age","response"])
# CATEGORICALS AND GROUPBY



df.groupby("treatment").mean()
df.groupby("treatment").age.max()
df.groupby("treatment")[["age","response"]].min()
data.groupby("Type 1").Attack.max()
df.info()

# as you can see gender is object

# However if we use groupby, we can convert it categorical data. 

# Because categorical data uses less memory, speed up operations like groupby

#df["gender"] = df["gender"].astype("category")

#df["treatment"] = df["treatment"].astype("category")

#df.info()