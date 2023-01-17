# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/pokemon-challenge/pokemon.csv")
data
data.info()
data.describe()
data.columns
data.corr()

#orantı tablosu verir.
f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt=".1f",ax=ax)

plt.show()
data.head(10)
#line plot

data.Speed.plot(kind="line",color="g",label="Speed",linewidth=1,alpha=0.5,grid=True,linestyle=":")

data.Defense.plot(color="r",label="Defense",linewidth=1,alpha=0.5,grid=True,linestyle="-.")

plt.legend(loc="upper right")

plt.xlabel("x axis",color="white")

plt.ylabel("y axis",color="white")

plt.title("Line plot",color="white")

plt.show()
#scatter plot

data.plot(kind="scatter",x="Attack",y="Defense",alpha=.5,color="r",grid=True,figsize=(10,10))

plt.xlabel("Attack",color="w")

plt.ylabel("Defense",color="w")

plt.title("Attack Defense Scatter Plot",color="w")

plt.scatter(data.Attack,data.Defense,alpha=.5,color="r")

plt.grid(True)

plt.show()
#histogram

# bins = number of bar in figure

data.Speed.plot(kind="hist",bins=50, figsize=(12,12))

plt.show()
# clf() = cleans it up again you can start a fresh

data.Speed.plot(kind = 'hist',bins = 50)

plt.clf()

# We cannot see plot due to clf()

#dictioanry

dictionary={"spain":"madrid","usa":"vegas"}

dictionary.keys()
print(dictionary.keys())
dictionary.keys()


print(dictionary.keys())
dictionary["spain"]="barcelona"

print(dictionary)
dictionary["france"]="paris"

dictionary
del dictionary["spain"]

dictionary
print("france"in dictionary)
dictionary.clear()

dictionary
data=pd.read_csv("/kaggle/input/pokemon-challenge/pokemon.csv")
series=data["Defense"]  # data['Defense'] = series

print(type(series))

data_frame=data[["Defense"]] # data[['Defense']] = data frame

print(type(data_frame))
# Comparison operator

print(3 > 2)

print(3!=2)

# Boolean operators

print(True and False)

print(True or False)
# 1 - Filtering Pandas data frame

x=data["Defense"]>200

data[x]
# 2 - Filtering pandas with logical_and

data[np.logical_and(data["Defense"]>200,data["Attack"]>100)]
# This is also same with previous code line. Therefore we can also use '&' for filtering.

data[(data["Defense"]>200)&(data["Attack"]>100)]
# Stay in loop if condition( i is not equal 5) is true

i=0

while i!=5:

    print("i is:",i)

    i+=1

print(i,' is equal to 5')
# Stay in loop if condition( i is not equal 5) is true

list1=[3,8,12,40,-5]

for i in list1:

    print('i is: ',i)

print('')
# Enumerate index and value of list

# index : value = 0:1, 1:2, 2:3, 3:4, 4:5

for index,value in enumerate(list1):

    print(index,":",value)

print('')

#for'dan sonra ilk yazılan index,daha sonra yazılan value içindir. 
# For dictionaries

# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part

dictionary = {'spain':'madrid','france':'paris'}

for key,value in dictionary.items():

    print(key," : ",value)

print('')

#for'dan sonra ilk yazılan key,daha sonra yazılan value içindir. 
# For pandas we can achieve index and value

for index,value in data[["Attack"]][0:1].iterrows():

    print(index," : ",value)
#USER DEFINED FUNCTION

# example of what we learn above

def tuple_ex():

    """return defined t tuple"""

    t=(1,2,3)

    return t

a,b,c=tuple_ex()

print("a:",a,"\nb:",b,"\nc:",c)
#SCOPE

x=2 #global x

def f():

    x=5  #local x

    return x

print(x) #global değeri döndürür.

print(f()) #tanımlanan fonksiyondaki local x'i döndürür.
#eğer local scope olmazsa;

x=2

def f():

    y=x*2  # there is no local scope x

    return y

print(f())   # it uses global scope x
# How can we learn what is built in scope

import builtins

dir(builtins)
#nested function

#iç içe fonksiyon

def square():

    """return square of value"""

    def add():

        """add two local variable"""

        x=2

        y=3

        z=x+y

        return z

    return add()**2

print(square())
# default arguments

#önceden tanımladığımız fonksiyonlar

def f(a,b=1,c=2):

    y=a+b+c

    return y

# eğer fonksiyonu döndürürken içine sadece 1 değer

# yazarsam onun a olduğunu anlar,zaten b ve c tanmlı

# ama eğer  değer girersem a,b ve c için o değerleri alır.

# sırayla alır,mesela a ve b için değer girip de c için 

# girmezsem c için default değeri alır,b ve a için 

# tanımlanan değeri alır.
f(5)
f(5,6)
f(5,6,7)
f(5,c=7)
# flexible arguments *args

def f(*args):

    for i in args:

        print(i)

f(1)

f("")

f(1,2,3,4)
def f(*args):

    a=0

    for i in args:

     a+=i

    return a

f(1,2)
f(1,2,3,4,5,6,7,8,9)
# flexible arguments **kwargs that is dictionary

def f(**kwargs):

    """print key and value of dictionary"""

    for key,value in kwargs.items():

# If you do not understand this part turn for loop part and look at dictionary in for loop

        print("key:",key,"value:",value)

f(country = 'spain', capital = 'madrid', population = 123456)
#LAMBDA FUNCTION

square=lambda x: x**2

print(square(3))
total=lambda a,b,c:a+b+c

print(total(3,-6,1))
#ANONYMOUS FUNCTİON

number_list=[1,2,3]

y=map(lambda x:x**2,number_list)

print(list(y))
name="marcelo"

it=iter(name)

print(next(it)) # print next iteration
print(*it) # print remaining iteration
#zip(): zip lists

#iki listeyi zip etmek(birleştirmek)

list1=[1,2,3,4]

list2=[5,6,7,8]

z=zip(list1,list2)

print(z)

z_list=list(z)

print(z_list)
#zip şeklindeki bir listeyi unzip ediyoruz. ==> zip(* ... )ile

un_zip=zip(*z_list)

unlist1,unlist2=list(un_zip) #burada list yada tuple diyebiliriz.

print(unlist1)               #ikisinde de tuple yapar.

print(unlist2)

print(type(unlist2))
#unzip yapınca bunları tuple yaptı, istersek liste yapabiliriz.

print(list(unlist1))

print(type(list(unlist1)))
#list comprehension

num1=[1,2,3]

num2=[i+1 for i in num1]

print(num2)
# Conditionals on iterable

num1 = [5,10,15]

num2=[i**2 if i==10 else i-5 if i<7 else i+5 for i in num1]

num2
# lets return pokemon csv and make one more list comprehension example

# lets classify pokemons whether they have high or low speed. Our threshold is average speed.

data=pd.read_csv("/kaggle/input/pokemon-challenge/pokemon.csv")

threshold = sum(data.Speed)/len(data.Speed)

data["speed_level"] = ["high" if i > threshold else "low" for i in data.Speed]

data.loc[:10,["speed_level","Speed"]] # we will learn loc more detailed later
#value_counts

print(data["Type 1"].value_counts(dropna=False)) # if there are nan values that also be counted

# As it can be seen below there are 112 water pokemon or 70 grass pokemon
data.describe()
data
data.boxplot(column="Attack",by="Legendary",figsize=(10,10))
#TIDY DATA

# Firstly I create new data from pokemons data to explain melt nore easily.

data_new = data.head()    # I only take 5 rows into new data

data_new
# lets melt

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted=pd.melt(frame=data_new,id_vars="Name",value_vars=["Attack","Defense"])

melted
#PIVOTING DATA

#Reverse of melting.
#CONCATENATING DATA

data1 = data.head()

data2= data.tail()

concat_data_row=pd.concat([data1,data2],axis=0,ignore_index=True)

concat_data_row
# ignore_index=True demezsek;

data1 = data.head()

data2= data.tail()

concat_data_row=pd.concat([data1,data2],axis=0)

concat_data_row
data["Type 1"]=data["Type 1"].astype("category")

data["Speed"]=data["Speed"].astype("float")
data.dtypes
#MISSING DATA and TESTING WITH ASSERT

data.info()

#type2 de 386 tane NaN varmış.
data["Type 2"].value_counts(dropna =False)

#dropna false diyerek nan ları da yazdırmasını söyledik.
# Lets drop nan values

data1=data.copy()

data1["Type 2"].dropna(inplace = True)

#inplace=True diyerek çıkardığın bu değerleri

#data1 içine kaydet demektir. Güncel hali

#data1 içine kaydetmiş olduk.

#PEKİ ÇALIŞTI MI?
# Assert statement:

assert 1==1 # return nothing because it is true
# assert 1==2 # return error because it is false

#eğer bunu döndürürsek hata verir çünkü yanlış.
assert  data1['Type 2'].notnull().all() # returns nothing because we drop nan values

#mesela bunu data için yapsak hata verirdi çünkü nan lar duruyor.
data1["Type 2"].fillna("empty",inplace=True)

#boş yerleri "empty" yazısı ile doldurduk
assert data1['Type 2'].notnull().all()  # returns nothing because we do not have nan values
assert data1.columns[1]=="Name"

#1. column Name mi diye kontrol ediyoruz. bir şey döndürmüyor,demek ki doğru.
assert data.Speed.dtypes == np.float
# data frames from dictionary

country=["Turkey","France","USA"]

population=["80","70","300"]

capital=["Ankara","Paris","Washington DC"]

list_label=["country","population","capital"]

list_col=[country,population,capital]

zipped=list(zip(list_label,list_col))

data_dict=dict(zipped)

df=pd.DataFrame(data_dict)

df
data_dict
# Add new columns

df["EU"]=["False","True","False"]

df
# Broadcasting

df["income"] = 0 #Broadcasting entire column

df
#VISUAL EXPLORATORY DATA ANALYSIS

# Plotting all data 

data2=data.loc[:,["Attack","Defense","Speed"]]

data2.plot()

# it is confusing
#subpolts

data2.plot(subplots=True)

plt.show()
# scatter plot  

data2.plot(kind="scatter",x="Attack",y="Defense")

plt.show()
# hist plot 

data2.plot(kind="hist")

#başka hibir şey girmezsek;
# hist plot  

data2.plot(kind="hist",y = "Defense",bins = 50,range=(0,250),normed = True)
# hist plot  

data2.plot(kind="hist",y = "Defense",bins = 50,range=(0,250))

#normed=True yazmazsak;
# hist plot  

data2.plot(kind="hist",y = "Defense",bins = 50)

#range vermzesek;
# hist plot  

data2.plot(kind="hist",y = "Defense",bins = 50,range=(0,1000))

#farklı bir range daha verelim;
# histogram subplot with non cumulative and cumulative

fig,axes=plt.subplots(nrows=2,ncols=1)

data2.plot(kind="hist",y="Defense",bins=50,range=(0,250),normed=True,ax=axes[0]) #non-cumulative

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative=True) #cumulative

plt.savefig('graph.png')

plt
# histogram subplot with non cumulative and cumulative

fig,axes=plt.subplots(nrows=4,ncols=1)

data2.plot(kind="hist",y="Defense",bins=50,range=(0,250),normed=True,ax=axes[0]) #non-cumulative

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative=True) #cumulative

plt.savefig('graph.png')

plt
data2.plot(kind="hist",y="Defense",bins=50,range=(0,250),normed=True,ax=axes[0]) #non-cumulative

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative=True) #cumulative

plt.savefig('graph.png')

plt

#ilk satırı yazmazsak
fig,at=plt.subplots(nrows=2,ncols=1)

data2.plot(kind="hist",y="Defense",bins=50,range=(0,250),normed=True,ax=at[0]) #non-cumulative

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = at[1],cumulative=True) #cumulative

plt.savefig('graph.png')

plt

#burada axes yazanların hepsine ax ta yazazbiliriz, at da. ne yazdığımı zönemli değil. bir şey değişmez.
fig,at=plt.subplots(nrows=2,ncols=1)

data2.plot(kind="hist",y="Defense",bins=50,range=(0,250),normed=True,ax=at[0]) #non-cumulative

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = at[1],cumulative=True) #cumulative

plt

# plt.savefig('graph.png') yazmasak da olur.
fig,axes=plt.subplots(nrows=2,ncols=1)

#data2.plot(kind="hist",y="Defense",bins=50,range=(0,250),normed=True,ax=axes[0]) #non-cumulative

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative=True) #cumulative

plt.savefig('graph.png')

plt
time_list=["1997-03-11","1997-10-22"]

print(type(time_list))

print(type(time_list[1]))

#biz string oluşturmuş olduk ama 

#biz datatime object istiyoruz.
datatime_object=pd.to_datetime(time_list)

print(type(datatime_object))
# close warning

import warnings

warnings.filterwarnings("ignore")

#yukardaki 3 satırın ne olduğu önemli değil, hata vermemesi için yazdık.

# In order to practice lets take head of pokemon data and add it a time list

data3=data.head()

data_list=["1991-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datatime_object=pd.to_datetime(data_list)

data3["date"]=datatime_object

data3
data3=data3.set_index("date")

data3

#index sütunumuz date'lerden oluşuyor.
# Now we can select according to our date index

print(data3.loc["1993-03-16"])
print(data3.loc["1992-03-10":"1993-03-16"])
data3.resample("A").mean()

#yılları mean'e göre resample ediyoruz.
data3.resample("M").mean()

#şimdi de ayları mean'e göre resample ediyoruz.
#data3.resample("M").first().interpolate("linear")

#sürekli hata veriyor,sorunu çözemedim.
data3.resample("M").mean().interpolate("linear")
#1st

# read data

data=pd.read_csv("/kaggle/input/pokemon-challenge/pokemon.csv")

data.head()
#2nd

#index changing

data=data.set_index("#")

data.head()
#3rd

# indexing using square brackets

data["Sp. Atk"][1]
data.HP[1]
data["HP"][1]
data.loc[1,["HP"]]
#4th

# Selecting only some columns

data[["HP","Attack"]]

#1st

# Difference between selecting columns: series and dataframes

print(type(data["HP"]))   # series

print(type(data[["HP"]])) # data frames
#2nd

# Slicing and indexing series

data.loc[1:10,"HP":"Defense"]
data.loc[1:10,["HP","Defense"]]
#3rd

# Reverse slicing 

data.loc[10:1:-1,"HP":"Defense"]
#4th

# From something to end

data.loc[1:10,"Speed":]

#Speed'den son column'a kadar tamamını aldık.
#1st

# Creating boolean series

boolean=data.HP>200

data[boolean]
boolean=data.HP>200

print(boolean)
boolean=data.HP>200

boolean
#2nd

# Combining filters

first_filter = data.HP > 150

second_filter = data.Speed > 35

data[first_filter & second_filter]
#3rd

# Filtering column based others

data.HP[data.Speed<15]
#1st

# def ile func tanımlayarak;

# Plain python functions

def div(n):

    return n/2

data.HP.apply(div)
#2nd

# apply lambda ile

# Or we can use lambda function

data.HP.apply(lambda n : n/2 )
#3rd

# Defining column using other columns

data["Total_Power"]=data.Attack + data.Defense

data.head()
data.Total_Power.head()
#1)Index'in ismini öğrenme:

print(data.index.name)
#2)İndex'in ismini değiştirme

data.index.name="index_name"

data.head()
#3)

# Overwrite index

# if we want to modify index we need to change all of them.

data.head()

# first copy of our data to data4 then change index 

data4 = data.copy()

# lets make index start from 100. It is not remarkable change but it is just example

data4.index=range(100,1700,2)

data4.head()
data=pd.read_csv("/kaggle/input/pokemon-challenge/pokemon.csv")

data.head()
#2 index column'ı oluşturma

# Setting index : type 1 is outer type 2 is inner index

data1=data.set_index(["Type 1","Type 2"])

data1.head(20)
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df = pd.DataFrame(dic)

df
# pivoting

df.pivot(index="treatment",columns = "gender",values="response")
df1=df.set_index(["treatment","gender"])

df1
# lets unstack it

df1.unstack(level=0)

#2 index column ı var. biz 0. index column ını siliyoruz.
df1.unstack(level=1)

#şimdi de 1. index column ını siliyoruz.
#,ndex column'larının yerini değiştirelim.

# change inner and outer level index position

df2=df1.swaplevel(0,1)

df2
df
#df.pivot(index="treatment",columns="gender",values="response")

pd.melt(df,id_vars="treatment",value_vars=["age","response"])

# biz bir şey girmediğimiz sürece variable ve value columnları otomatik olarak gelir.
# We will use df

df
df.groupby("treatment").mean()  # mean is aggregation / reduction method

# there are other methods like sum, std,max or min
# we can only choose one of the feature

df.groupby("treatment").age.max()
df.groupby("treatment").max()
df.groupby("treatment")[["age","response"]].min()
df.groupby("treatment").min()