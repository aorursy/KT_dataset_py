# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/pokemon.csv")
data.info()
data.describe()
data.corr() #Dataki bütün correlationları küçükten büyüğe sıralar.
#Dataki bütün correlationları küçükten büyüğe sıralar.

data.corr().unstack().sort_values().drop_duplicates()
#correlation map

plt.subplots(figsize=(10,10))

sns.heatmap(data.corr(),annot=True,linewidth=".5",cmap="YlGnBu",fmt=".1f")

plt.show()

#figsize - plotun boyutlarını 

#data.corr() - feature lar arasındaki ilişkiyi gösterir

#annot=True -correlation oranlarını gösterir

#linewidths - aralardaki line ların kalınlığını belirler

#cmap - kullanacağımız renk tonlarını belirler

#fmt - precision(0'dan sonraki basamak sayısı)'ı belirler

#eğer iki feature arasındaki correlation 1 veya 1'e yakın ise iki feature arasındaki correlation da doğru(pozitif) orantı vardır.

#eğer iki feature arasındaki correlation -1 veya -1'e yakın ise iki feature arasındaki correlation da ters(negatif) orantı vardır.

#eğer 0 veya 0'a yakın çıkarsa aralarında ilişki yoktur.
data.head() #ilk beş satır
data.tail() #son beş satır
data.sample(5) #rastgele beş satır
data.columns
data.dtypes
data.drop('#', axis = 1, inplace = True) # gereksiz sütunu çıkaralım
data.isnull() #false lar değer olduğunu true lar olmadığını gösterir
data.isnull().sum() #Datamız içerisinde tanımlanmamış değerler 
data.isnull().sum().sum()  #Datamız içerisinde toplam tanımlanmamış değerler 
data[["Name"]].isnull()
data.sort_values("HP").head(20)
data.sort_values("HP", ascending=False).head()
data2 = data[["Name", "HP"]].head()

data2
columnsRequired = ["Name", "HP"]

data3 = data[columnsRequired].head()

data3
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.Speed.plot(kind="line", color="g", label="Speed", linewidth=1, alpha=0.5, grid=True, linestyle=":", figsize=(12,12))

data.Defense.plot(color="r", label="Defense", linewidth=1, alpha=0.5, grid=True, linestyle="-.")

plt.legend(loc="upper right")# legend = puts label into plot

plt.xlabel("x axis")         # label = name of label

plt.ylabel("y axis")

plt.title("line Plot")       # title = title of plot



#plt.xticks(np.arange(ilk değer,son değer,step)) 

plt.xticks(np.arange(0,800,30)) #x eksenindeki değerlerin aralıklarını belirler

plt.yticks(np.arange(0,300,30)) #y eksenindeki değerlerin aralıklarını belirler

plt.show()
# Scatter Plot 

# x = attack, y = defense

data.plot(kind="scatter", x="Attack", y="Defense", alpha=0.5, color="red", figsize=(5,5))

plt.xlabel("Attack")    # label = name of label

plt.ylabel("Defense")

plt.title("Attack Defense Scatter Plot") # title = title of plot

plt.show()
# Histogram

# bins = number of bar in figure

data.Speed.plot(kind="hist", bins=50, figsize=(10,10))

plt.show()

#bins - bar sayısını belirler
# clf() = cleans it up again you can start a fresh

data.Speed.plot(kind="hist", bins=50)

plt.clf() # We can not see plot if we use clf() method
#we dont use.its just example

dic2 = [{"id": 825, "name": "support group"}, {"id": 851, "name": "dual identity"}]

df2 = pd.DataFrame(dic2)

df2
#create dictionary and look its keys and values

dictionary = {"spain":"madrid","usa":"vegas"}

print(dictionary.keys())

print(dictionary.values())
# Keys have to be immutable objects like string, boolean, float, integer or tubles

# List is not immutable

# Keys are unique

dictionary["spain"] = "barcelona" # update existing entry

print(dictionary)

dictionary["france"] = "paris"    #Add new entry

print(dictionary)

del dictionary["spain"]           # remove entry with key 'spain'

print(dictionary)

print("france" in dictionary)     # check include or not

dictionary.clear()                # remove all entries in dict

print(dictionary)
# In order to run all code you need to take comment this line

#del dictionary         # delete entire dictionary     

print(dictionary)       # it gives error because dictionary is deleted
print(type(data)) # pandas.core.frame.DataFrame

print(type(data["Defense"])) #pandas.core.series.Series

print(type(data["Defense"].values)) #numpy.ndarray
series = data['Defense']        # data['Defense'] = series

data_frame = data[['Defense']]  # data[['Defense']] = data frame



print(type(series))

print(type(data_frame))



print(series)

data_frame
# Comparison operator

print(3 > 2)

print(3!=2)

# Boolean operators

print(True and False)

print(True or False)
# 1 - Filtering Pandas data frame

x = data["Defense"]>200     # There are only 3 pokemons who have higher defense value than 200

data[x]
# 2 - Filtering pandas with logical_and

# There are only 2 pokemons who have higher defence value than 200 and higher attack value than 100

data[np.logical_and(data["Defense"]>200,data["Attack"]>100)]
# This is also same with previous code line. Therefore we can also use '&' for filtering.

data[(data["Defense"]>200) & (data["Attack"]>100)]
# Stay in loop if condition( i is not equal 5) is true

i = 0

while i != 5:

    print("i is: ",i)

    i+=1

print(i," is equal to 5")
# Stay in loop if condition( i is not equal 5) is true

lis = [1,2,3,4,5]



for i in lis:

    print("i is: ",i)

print("")    

# Enumerate index and value of list

# index : value = 0:1, 1:2, 2:3, 3:4, 4:5

for index,value in enumerate(lis):

    print(index," : ",value)

print("")

# For dictionaries

# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.

dictionary = dictionary = {'spain':'madrid','france':'paris'}

for key,value in dictionary.items():

    print(key," : ",value)

print("")

# For pandas we can achieve index and value

for index,value in data[["Attack"]][0:5].iterrows():

    print(index," : ",value)

data[["Attack"]][0:5]
# example of what we learn above

def tuple_ex():

    """ return defined t tuble"""

    t = (1,2,3)

    return t

a,b,c = tuple_ex()

print(a,b,c)
# guess print what

x = 2

def f():

    x=3

    return x

print(x)      # x = 2 global scope

print(f())    # x = 3 local scope
# What if there is no local scope

x = 5

def f():

    y = 2*x        # there is no local scope x

    return y

print(f())         # it uses global scope x

# First local scopesearched, then global scope searched, if two of them cannot be found lastly built in scope searched.
# How can we learn what is built in scope

import builtins

dir(builtins)
#nested function

def square():

    """ return square of value """

    def add():

        """ add two local variable """

        x = 2

        y = 3

        z = x + y

        return z

    return add()**2

print(square())    
# default arguments

def f(a, b = 1, c = 2):

    y = a + b + c

    return y

print(f(5))

# what if we want to change default arguments

print(f(5,4,3))
# flexible arguments *args

def f(*args):

    for i in args:

        print(i)

f(1,1,2)

print("")

f(1,2,3,4)

print("")

f("orhan","kadir","cemal",1)

# flexible arguments **kwargs that is dictionary

def f(**kwargs):

    """ print key and value of dictionary"""

    for key, value in kwargs.items():               # If you do not understand this part turn for loop part and look at dictionary in for loop

        print(key, " ", value)

f(country = 'spain', capital = 'madrid', population = 123456)
# lambda function

square = lambda x: x**2     # where x is name of argument

print(square(4))

tot = lambda x,y,z: x+y+z   # where x,y,z are names of arguments

print(tot(1,2,3))
number_list=(1,2,3,4,5,6,7,8,9)

y = map(lambda x : x**2 ,number_list)

#liste_Y = list(y) #map data tipini list'e dönüştürdük

#print(liste_Y) #[1, 4, 9, 16, 25, 36, 49, 64, 81]

#OR short way

print(list(y)) #[1, 4, 9, 16, 25, 36, 49, 64, 81]
# iteration example

name = "Orhan"

itr = iter(name)

print(next(itr))# print next iteration

print(next(itr))# print next iteration

print(*itr)     # print remaining iteration
list1 = [1,2,3,4]

list2 = [5,6,7,8]

z = zip(list1,list2)

print(z)

z_list = list(z)  #converting zip to list type

print(z_list)

print("")    

itr = iter(z_list) 

print(next(itr))   # print next iteration

print(*itr)        # print remaining iteration
un_zip = zip(*z_list)

unlist1,unlist2 = list(un_zip) # unzip returns tuple

print(unlist1)

print(unlist2)

print(type(unlist1))

print(type(list(unlist1))) #if we want to change data type tuple to list we need to use list() method.
num1 = [1,2,3]

num2 = [i+1 for i in num1]

print(num2)

#OR

print([i+1 for i in num1])
# Conditionals on iterable

num1 = [5,10,15]

num2 = [i**2 if i==10 else i-5 if i<7 else i+5 for i in num1]

print(num2)
# lets return pokemon csv and make one more list comprehension example

# lets classify pokemons whether they have high or low speed. Our threshold is average speed.

threshold = sum(data.Speed)/len(data.Speed)

data["speed_level"] = ["high" if i>threshold else "low" for i in data.Speed]

data.loc[:10,["speed_level","Speed"]]
data = pd.read_csv('../input/pokemon.csv')

data.head()  # head shows first 5 rows
#örneğin columnlar arasında type 1 adında bir feature var.Bunu clean etmemiz gerekir.

#datadaki bu unclean şeyleri düzeltmemiz gerekir

#data.Type 1 diyemeyiz

#data["Type 1"] diye çağırabiliriz

#Bu gibi durumları ortadan kaldırmak için columnları belirli bir formata(lowercase-upper case) getirmek gerekir.

#data.type1 haline getirmek daha uygun olacaktır.

#data.columns = [each.lower() for each in data.columns]

#print(data.columns)

#***OR

#data.columns = [each.upper() for each in data.columns]

#print(data.columns)



#dataFrame1.columns = [each.split()[0]+"_"+each.split()[1] if(len(each.split())>1) else each for each in dataFrame1.columns]

#print(dataFrame1.columns)

#***OR birden fazla boşluk için

#data.columns = [each.replace(" ","_") if(len(each.split())>1) else each for each in data.columns]

#print(data.columns)   
# tail shows last 5 rows

data.tail()
# columns gives column names of features

data.columns
# shape gives number of rows and columns in a tuple

data.shape
# info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage

data.info()
data.dtypes
data["Type 1"].unique()
data["Type 1"] == "Dragon" # We can filter the data if we want 

data[data["Type 1"] == "Dragon"]
# For example lets look frequency of pokemom types

print(data["Type 1"].value_counts(dropna=False,sort=True))# if there are nan values that also be counted

#sort : boolean, default True   =>Sort by values

#dropna : boolean, default True =>Don’t include counts of NaN.

# As it can be seen below there are 112 water pokemon or 70 grass pokemon
# Lets check Type 2

print(data["Type 2"].value_counts(dropna=False,sort=True))

# As you can see, there are 386 NAN values
# For example max HP is 255 or min defense is 5

data.describe() #ignore null entries
# For example: compare attack of pokemons that are legendary  or not

# Black line at top is max

# Blue line at top is 75%

# Red line is median (50%)

# Blue line at bottom is 25%

# Black line at bottom is min

#Outlier are smaller than Q1 - 1.5(Q3-Q1) and bigger than Q3 + 1.5(Q3-Q1).     (Q3-Q1) = IQR

data.boxplot(column='Attack',by = 'Legendary',fontsize=12,figsize=(14,14))



data2 = data[data["Legendary"]==False]

print(data2.Attack.max())

print(data2.Attack.quantile(q=0.75))

print(data2.Attack.quantile(q=0.5))

print(data2.Attack.quantile(q=0.25))

print(data2.Attack.min())



data3 = data[data["Legendary"]==True]

print(data3.Attack.max())

print(data3.Attack.quantile(q=0.75))

print(data3.Attack.quantile(q=0.5))

print(data3.Attack.quantile(q=0.25))

print(data3.Attack.min())



# FINDING OUTLIERS

#Formülü dataya göre değiştirmeliyiz



#FOR data2

print([x for x in data2.Attack if x<(data2.Attack.quantile(0.25)-1.5*(data2.Attack.quantile(0.75)-data2.Attack.quantile(0.25)))])

print([x for x in data2.Attack if x>(data2.Attack.quantile(0.75)+1.5*(data2.Attack.quantile(0.75)-data2.Attack.quantile(0.25)))])



#FOR data3

print([x for x in data3.Attack if x<(data3.Attack.quantile(0.25)-1.5*(data3.Attack.quantile(0.75)-data3.Attack.quantile(0.25)))])

print([x for x in data3.Attack if x>(data3.Attack.quantile(0.75)+1.5*(data3.Attack.quantile(0.75)-data3.Attack.quantile(0.25)))])
#Outlier are smaller than Q1 - 1.5(Q3-Q1) and bigger than Q3 + 1.5(Q3-Q1). (Q3-Q1) = IQR



#bütün columnları ve rowları gösterir.

pd.set_option("display.max_columns",None) 

pd.set_option("display.max_rows",None)





data.boxplot(column ="Defense",by="Type 2",grid=True,fontsize=12,figsize=(14,14))



#2 örneği inceleyelim ve boxplot'tan doğrulayalım.

data2 = data[data["Type 2"]=="Ground"]



data3 = data[data["Type 2"]=="Rock"]



#for data2

print(data2.Defense.max())

print(data2.Defense.quantile(q=0.5)) #q=quantile

print(data2.Defense.quantile(q=0.25))

print(data2.Defense.quantile(q=0.75))

print(data2.Defense.min())

print(data2.std())

#for data3

print(data3.Defense.max())

print(data3.Defense.quantile(q=0.5)) #q=quantile

print(data3.Defense.quantile(q=0.25))

print(data3.Defense.quantile(q=0.75))

print(data3.Defense.min())

print(data3.std())



# FINDING OUTLIERS

#Formülü dataya göre değiştirmeliyiz

data4 = data[data["Type 2"]=="Psychic"]

for x in data4.Defense:

    if x<(data4.Defense.quantile(0.25)-1.5*(data4.Defense.quantile(0.75)-data4.Defense.quantile(0.25))):

       print(x)

#doing with list comprehension

#burada else kullansaydık if 'i başa alırdık ama eğer kullanmıyacaksak başa alırsak else kullanmamız için bize hata verir

#o yüzden if i sona almalıyız

print([x for x in data4.Defense if x<(data4.Defense.quantile(0.25)-1.5*(data4.Defense.quantile(0.75)-data4.Defense.quantile(0.25)))])

print([x for x in data4.Defense if x>(data4.Defense.quantile(0.75)+1.5*(data4.Defense.quantile(0.75)-data4.Defense.quantile(0.25)))])
# Firstly I create new data from pokemons data to explain melt more easily.

data_new = data.head()    # I only take 5 rows into new data

data_new
# lets melt

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted = pd.melt(frame=data_new, id_vars = "Name",value_vars=["Attack","Defense"])

melted
# Index is name

# I want to make that columns are variable

# Finally values in columns are value

melted.pivot(index="Name", columns = "variable", values="value")
#vertical olarak birlestir

data1 = data.head()

data2 = data.head()

v_concat = pd.concat([data1,data2],axis=0,ignore_index=True)

v_concat
# Firstly lets create 2 data frame

data1 = data.Attack.head()

data2 = data.Defense.head()

h_concat = pd.concat([data1,data2], axis=1)

h_concat
hp = data.HP.head()

attack = data.Attack.head()

h_concat = pd.concat([hp,attack,hp],axis=1)

h_concat
name = data.Name.head()

type1 = data["Type 1"].head()

hp = data.HP.head()

attack = data.Attack.head()

h_concat = pd.concat([name+" - "+type1,hp*1.8,attack],axis=1)

h_concat
data.dtypes
# lets convert object(str) to categorical and int to float.

#DONT forget ,Setting return back default setting to int

#data["Type 1"] = data["Type 1"].astype("category")

#data.Speed = data.Speed.astype("float")

#data.Speed[0:10] #as you see it is converted from int to float
# As you can see Type 1 is converted from object to categorical

# And Speed is converted from int to float

data.dtypes
# Lets look at does pokemon data have nan value

# As you can see there are 800 entries. However Type 2 has 414 non-null object so it has 386 null object.

data["Type 2"][4]
# Lets chech Type 2

data["Type 2"].value_counts(dropna =False)

# As you can see, there are 386 NAN value
# Lets drop nan values

# also we will use data to fill missing value

data["Type 2"].dropna(inplace = True)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data

data
#  Lets check with assert statement

# Assert statement:

assert 1==1 # return nothing because it is true

data
#In order to run all code, we need to make this line comment

#assert 1==2 # return error because it is false
assert  data['Type 2'].notnull().all() # returns nothing because we drop nan values
data["Type 2"].fillna("empty",inplace = True)

data
assert  data['Type 2'].notnull().all() # returns nothing because we do not have nan values
# # With assert statement we can check a lot of thing. For example

assert data.columns[1] == 'Name'

assert data.Speed.dtype == np.int

#OR

assert data.Speed.dtype == "int64"

print(data.Speed.dtypes)
country = ["Spain","France"]

population = ["1000","2000"]

list_label = ["country","population"]

list_col = [country,population]

print(list_col)

zipped = list(zip(list_label,list_col))

print(zipped)

data_dict = dict(zipped)

print(data_dict)

df = pd.DataFrame(data_dict)

df
df["capital"]=["madrid","paris"]

df
df["income"] = 0

df
# Plotting all data 

data1 = data.loc[:,["Attack","Defense","Speed"]]

data1.plot()

# SAME THING

#data.Attack.plot()

#data.Defense.plot()

#data.Speed.plot()
# subplots

data1.plot(subplots = True)

plt.show()
# scatter plot  

data1.plot(kind = "scatter",x="Attack",y = "Defense")

plt.show()
# close warning

import warnings

warnings.filterwarnings("ignore")



# hist plot  

data1.Defense.plot(kind = "hist",bins = 50,range= (0,250),normed = True)

plt.show()
# histogram subplot with non cumulative and cumulative

fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[0])

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt
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

#OR

#data2.set_index("date",inplace=True)

data2 
#bütün columnları ve rowları gösterir.

pd.set_option("display.max_columns",None) 

pd.set_option("display.max_rows",None)



# Now we can select according to our date index

print(data2.loc["1993-03-16"]) #print(data2.loc["1993-03-16",:]) same thing

print(data2.loc["1992-03-10":"1993-03-16"])
# We will use data2 that we create at previous part

data2.resample("A").mean() #yıldan yıla featureların kendi içinde ortalaması
# Lets resample with month

data2.resample("M").mean()

# As you can see there are a lot of nan because data2 does not include all months
# In real life (data is real. Not created from us like data2) we can solve this problem with interpolate

# We can interpolete from first value

data2.resample("M").first().interpolate("linear")
# Or we can interpolate with mean()

data2.resample("M").mean().interpolate("linear")
data = pd.read_csv("../input/pokemon.csv")

data.set_index("#",inplace=True)

data.head()
# indexing using square brackets

data["HP"][1]
# using column attribute and row label

data.HP[1]
# using loc accessor

data.loc[2,["HP"]]
# Selecting only some columns

data[["HP"]]
# Difference between selecting columns: series and dataframes

print(type(data["HP"]))     # series

print(type(data[["HP"]]))   # data frames
# Slicing and indexing series

data.loc[1:10,"HP":"Defense"]   # 10 and "Defense" are inclusive
# Reverse slicing 

a =data.loc[10:1:-1,"Defense":"HP":-1] 

a
# From something to end

data.loc[1:10,"Speed":] 
# Creating boolean series

boolean = data.HP > 200

data[boolean]
# Combining filters

first_filter = data.HP > 150

second_filter = data.Speed > 35

data[np.logical_and(first_filter,second_filter)]

#OR

#data[np.logical_and(first_filter,second_filter)]
# Filtering column based others

data.HP[data.Speed<15]
# Filtering column based others

data[["HP"]][data.Speed<15]
# Filtering column based others

a = data[data.Speed<15]

a[["HP"]]
# Plain python functions

def div(n):

    return n/2

data["new_hp"]=data.HP.apply(div)

data
data["new_hp"] = data.HP.apply(lambda hp : hp/2)

data
# Defining column using other columns

data["total_power"] = data.Attack + data.Defense

data.head()
# our index name is this:

print(data.index.name)

#lets change it

data.index.name = "index_name"

data.head()
# Overwrite index

# if we want to modify index we need to change all of them.

data.head()

# first copy of our data to data3 then change index

data3 = data.copy()

# lets make index start from 100. It is not remarkable change but it is just example

data3.index = range(0,800,1)#800 exclusive->799

data3.head()
# We can make one of the column as index. I actually did it at the beginning of manipulating data frames with pandas section

# It was like this

# data= data.set_index("#")

# also you can use 

data3.index = data["#"]

data3.index = data["Name"]

data3.index = data["#"]

data3.head()

#set_index kullanırsak o feature ı index yapar bir daha feature yapmayız

#ama data["#"] series şeklinde verirsek hem column hem feature olarak kullanabiliriz.
# lets read data frame one more time to start from beginning

data = pd.read_csv('../input/pokemon.csv')

data.head()

# As you can see there is index. However we want to set one or more column to be index
# Setting index : type 1 is outer type 2 is inner index

data1 = data.set_index(["Type 1","Type 2"]) 

data1

# data1.loc["Fire","Flying"] # how to use indexes
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df = pd.DataFrame(dic)

df
# pivoting

df.pivot(index="treatment",columns = "gender",values="response")
df1 = df.set_index(["treatment","gender"])

df1

#OR

#df1 = df.set_index(["gender","treatment"])

# lets unstack it
# level determines indexes

df1.unstack(level=0)
df1.unstack(level=1)
# change inner and outer level index position

df2 = df1.swaplevel(0,1)

df2
df
pd.melt(df,id_vars="treatment",value_vars=["age","response"])
data.dropna(inplace=True) #NaN olan satırlar silinir
data.groupby("Type 2").Name.count() #type2 ye göre sırala ve o type da kaç tane pokemon ismi var
data.groupby("Type 2").Name.count().sum() #type2 ye göre sırala ve o type da kaç tane pokemon ismi var
data.groupby("Type 2").Name.count().sort_values(ascending=False).head(10)
data.groupby("Type 2").Name.count().sort_values(ascending=False).head(10).plot(kind="line")

#let find flying counts

data[data["Type 2"]=="Flying"]["Type 2"].count()
data.groupby("Type 2").Name.count().sort_values(ascending=False).head(10).plot(kind="bar")
data.groupby("Type 2").Name.count().head(10).plot(kind="bar")
data.groupby("Type 2").Name.count().plot(kind="bar")
data.groupby("Type 2").Name.count().sort_values(ascending=False).head(10).plot(kind="hist",bins=20)
data.groupby("Type 2").Name.count().sort_values(ascending=False).head(10).plot(kind="box")
data.groupby("Type 2").Name.count().sort_values(ascending=False).head(10).plot(kind="area")
data.groupby("Type 2").Name.count().sort_values(ascending=False).head(10).plot(kind="pie")
# We will use df

df
# according to treatment take means of other features

df.groupby("treatment").mean()   # mean is aggregation / reduction method

# there are other methods like sum, std,max or min
# we can only choose one of the feature

df.groupby("treatment").age.mean() 

#OR

#df.groupby("treatment")[["age"]].mean() 
df.groupby("treatment").mean().sort_values("age",ascending=False)
# we can only choose one of the feature

df.groupby("treatment").age.max()
# Or we can choose multiple features

df.groupby("treatment")[["age","response"]].mean() #dataframe olarak gösterir

#OR

#df[["age"]].mean() #series olarak gösterir
df.groupby("treatment")[["age"]].mean() 
df.info()

# as you can see gender is object

# However if we use groupby, we can convert it categorical data. 

# Because categorical data uses less memory, speed up operations like groupby

#df["gender"] = df["gender"].astype("category")

#df["treatment"] = df["treatment"].astype("category")

#df.info()
