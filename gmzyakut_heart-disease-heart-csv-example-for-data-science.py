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
heart_data = pd.read_csv("../input/heart.csv")
heart_data.info()
heart_data.corr()
#correlation map

f,axx= plt.subplots(figsize=(10,10))

sns.heatmap(heart_data.corr(), annot=True, linewidths=.5, fmt=".2f", ax=axx)

plt.show()
heart_data.head()
heart_data.columns
#Line Plot



heart_data.trestbps.plot(kind="line", color="g", label="resting blood pressure",linewidth=1,alpha = 1,grid = True,linestyle = ':')

heart_data.thalach.plot(kind="line", color="r", label="maximum heart rate achieved", linewidth=1,alpha = 1,grid = True,linestyle = ':')

plt.legend(loc='upper right')    

plt.xlabel('x axis')              

plt.ylabel('y axis')

plt.title('Line Plot')            

plt.show()

# Scatter Plot



heart_data.plot(kind="scatter", x="trestbps", y="age", alpha=0.8, color="r")

plt.xlabel("Resting blood pressure")

plt.ylabel("Age")

plt.title("Resting blood pressure - Age")
#Histogram



heart_data.chol.plot(kind = "hist", bins = 50, figsize = (15,15))

plt.xlabel("serum cholesterol in mg/dl")

plt.show()
#clf



heart_data.chol.plot(kind = "hist", bins = 50)

plt.clf()
dictionary = {"pencil" : "kalem", "table" : "masa"}

print(dictionary.keys())

print(dictionary.values())
dictionary["pencil"] = "kurşun kalem"

print(dictionary)



dictionary["book"] = "kitap"

print(dictionary)



del dictionary["pencil"]

print(dictionary)



print("table" in dictionary)



dictionary.clear()

print(dictionary)
heart_data = pd.read_csv("../input/heart.csv")
series = heart_data["thalach"]

print(type(series))

print(series.head())

print("-----------")

data_frame = heart_data[["thalach"]]

print(type(data_frame))

print(data_frame.head())
# Comprasion operator

print(3>2)

print(3!=2)

print("-----------")

#Boolean operators

print(True and False)

print(True or False)
#Filtering



a = heart_data["thalach"]>180 # return True or False 



heart_data[a] # return table of this values
# filtering logical_and

# "thalach">180 and "chol">180



heart_data[np.logical_and(heart_data["chol"]>250, heart_data["thalach"]>180)]
# filtering &



heart_data[(heart_data["chol"]>250) & (heart_data["thalach"]>180)]
# while

i = 0

while i != 5 :

    print("i is: ", i)

    i += 1

print(i, "is equal to 5")
# for

lis = [1,2,3,4,5]

for i in lis:

    print("i is:", i)

print(" ")
# Enumerate index and value of lis

# index : value = 0 : 1, 1 : 2 .....

for index, value in enumerate(lis):

    print(index, " : ", value)
# for dictionary

dictionary = {"spain" : "madrid", "france" : "paris"}

for key, value in dictionary.items():

    print(key, " : ", value)
# Pandas index and value

for index, value in heart_data[["chol"]][0:3].iterrows():

    print(index, " : ", value)
#example of what we learn above

def tuble_ex():

    """docstring example"""

    """return defined tuble"""

    t=(1,2,3)

    return t

a,b,c = tuble_ex()

print(a,b,c)
# global and local scope

x = 5

def f():

    x = 10

    return x

print(x) # global

print(f()) #local
# what if there is no local scope

# local değişkenin olmadığı yerde global değişken kullanılır

x = 5

def f():

    y = x * 10

    return y

print(f())
# how can we learn what is built in scope

import builtins

dir(builtins)
# nested function

def square():

    """return square of value"""

    def add():

        """add two local variables"""

        x = 3

        y = 4

        z = x + y

        return z

    return add()**2 #add fonsiyonu ile dönen değerin karesidir

print(square())
#Default argument

def f(a, b=2, c=3):

    return a*b*c



print(f(5)) # b and c are default arguments



print(f(2,4,6))
# flexible argument

#def f(*args):

def f(*args):

    for i in args:

        print(i)

f(1)

print("-----")

f(1,2,3,4)
# **kwargs is dictionary 

def f(**kwargs):

    """print key and value of dictionary"""

    for key,value in kwargs.items():

        print(key, ": ", value)



f(country='spain', capital='madrid', population='123456')
# user defined function (long way)

def square(x):

    return x**2

print(square(5))



# lambda function (short way)

square = lambda x : x **2

print(square(4))



tot = lambda x,y,z : x+y+z

print(tot(3,4,5))
# map(func, seq)



number_list = [1,2,3]

y = map(lambda x: x*5, number_list)

print(list(y))
#iteration example

name="python"

it = iter(name)

print(next(it))

print(next(it))

print(next(it))

print(*it)
# zip example

list1 = [1,2,3,4]

list2 = [5,6,7,8]



z = zip(list1,list2)

print(z)



z_list=list(z)

print(z_list)
# unzip example

un_zip = zip(*z_list)



un_list1, un_list2 = list(un_zip)



print(un_list1)

print(un_list2)



print(type(un_list1))

print(type(list(un_list2)))
#example of list comprehension

num1 = [1,2,3]

num2 = [i+1 for i in num1]



print(num2)
# conditionals on iterable

num1 = [5, 10, 15]

num2 = [i**2 if i==10 else i-5 if i<7 else i+5 for i in num1]

print(num2)
# list comprehension - heart.csv



ave_age = sum(heart_data.age) / len(heart_data.age)

print("average of age: ", ave_age)



heart_data["average_age"] = ["high" if i>ave_age else "down" for i in heart_data.age]



heart_data.loc[:10,["average_age","age"]]
# head shows first 5 rows

heart_data.head() 
# tail

heart_data.tail()
# columns gives column names of features

heart_data.columns
# shape gives number of rows and columns in a tuble

heart_data.shape
# info gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage

heart_data.info()
# frequency of data

print(heart_data["age"].value_counts(dropna=False))

#print(heart_data.age.value_counts(dropna=False))
# describe of data

heart_data.describe()
# box plot

heart_data.boxplot(column="trestbps", by="sex")

plt.show()



heart_data.boxplot(column="chol", by="age")

plt.show()



heart_data.boxplot(column="trestbps", by="age")

plt.show()
# new dataset

heart_data_new = heart_data.head(10)

heart_data_new
# melt()

# id_vars: hangi sütuna göre tablo hazırlayacağız

# value_vars: id_vars ın hangi değerlerini alacağız

melted = pd.melt(frame=heart_data_new, id_vars="age", value_vars=["chol","thalach"])

melted
# pivot

#melted.pivot(index="age", columns="variable", values="value")

melted.pivot_table(index = "age", columns = "variable", values = "value")
# two dataframe create

data1 = heart_data.head()

data2 = heart_data.tail()

conc_heart = pd.concat([data1,data2],axis=0, ignore_index=True)

#ignore_index - True:Yeniden index numarası verir False: kendi index numaralarını gösterir

#axis - 0: dikey olarak alt alta birleştirir 1: yatay olarak yan yana birleştirir

conc_heart
# axis = 1

data1 = heart_data.trestbps.head()

data2 = heart_data["thalach"].head()

conc_heart_col = pd.concat([data1,data2], axis = 1)

conc_heart_col
# data type

heart_data.dtypes
# Lets convert object (str) to category and int to float

heart_data.target = heart_data.target.astype("float")

heart_data.average_age = heart_data.average_age.astype("category")



heart_data.dtypes

heart_data.info()
# dropna = False nan olanları da listede gösterir

heart_data.age.value_counts(dropna = False) # NaN değer olmadığı için listede göstermedi
data1 = heart_data # NaN olan değer varsa listeden çıkaracağımız için data1 oluşturduk



data1.age.dropna(inplace = True) # age sütununda NaN olanları listeler, listeden çıkarır, son halini data1 içne kaydeder
# Kontrol için assert kullanılır. doğru ise boş döner

assert 1==1
# hatalı ise hata verir

#assert 1==2
# age sütunu içinde boş olmayanları

assert heart_data.age.notnull().all() # boş olmayanların hepsini listele = doğru mu doğru. o zaman boş döner
# eğer NaN değer varsa empty olarak doldur

heart_data.age.fillna("empty", inplace = True)
assert heart_data.age.notnull().all() # tekrar boş olanları kontrol ettik
#datasetin 1 sütunun adı sex doğru ise boş döndürür

assert heart_data.columns[1] == "sex"
#datasetin oldpeak sütunun tipi float ise boş dner

assert heart_data.oldpeak.dtypes == np.float
# data frames from dictionary

country = ["Spain", "France"]

population = ["11", "12"]

list_label = ["country", "populatin"]

list_col = [country, population]

zipped = list(zip(list_label, list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
# add new columns

df["capital"] = ["Madrid", "Paris"]

df
# broadcasting

df["income"] = 0

df
# plotting all data

heart_data.loc[:, ["chol", "thalach"]]

data1.plot()
#subplot

data1.plot(subplots= True)

plt.show()
#scatter plot

data1.plot(kind="scatter", x="chol", y="thalach")

plt.show()
# hist plot

data1.plot(kind="hist", y="chol", bins=50,range=(0,500), normed=True)
#histogram subplot with non cumulative and cumulative

fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "chol",bins = 50,range= (0,250),normed = True,ax = axes[0])

data1.plot(kind = "hist",y = "chol",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt
heart_data.describe()
time_list = ["1992-03-08","1992-04-12"]

print(type(time_list[1])) # As you can see date is string



# datetime_object değişkenine datetime tipinde aktarırız

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
# close warning

import warnings

warnings.filterwarnings("ignore")
# datetime uygulamak için heart datanın ilk 5 satırı alınıp date sütunu eklemesi yapılacak

data2 = heart_data.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object

# lets make date as index

data2= data2.set_index("date")

data2 
# Now we can select according to our date index

print(data2.loc["1993-03-16"])

print(data2.loc["1992-03-10":"1993-03-16"])
# We will use data2 that we create at previous part

data2.resample("A").mean()
# M = month

data2.resample("M").mean()
# intw-erpolate

data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")
# read data

heart_data = pd.read_csv("../input/heart.csv")

heart_data = heart_data.set_index("age")

heart_data.head()
# indexing using square brackets

# chol sütunundaki 40 yaşında olan kayıtları listeler

heart_data["chol"][40]
# using column attribute and row label

heart_data.chol[40]
# using loc accessor

# set_index (age) i 40 olan kayıpların chol değerlerini verir 

heart_data.loc[40,["chol"]]
# Selecting only some columns

heart_data[["chol","thalach"]]
# Difference between selecting columns: series and dataframes

print(type(heart_data["chol"]))     # series

print(type(heart_data[["chol"]]))   # data frames
#slicing and indexing series

# set_index age ayarlı olduğu için aralık şeklinde yazamadık. 

# her yaş değerinden fazlaca olduğu için sadece tek yaş değeri girebiliriz.

heart_data.loc[40,"chol":"thalach"]
# csv'yi tekrar okuttukdan sonra aralık şeklinde girebiliriz.

heart_data = pd.read_csv("../input/heart.csv")

# 0'dan 10'a kadar chol ile thalach arasındaki sütunları listeler

heart_data.loc[0:10,"chol":"thalach"]
#reverse slicing

heart_data.loc[10:0:-1,"chol":"thalach"]
# from something to end

heart_data.loc[1:10,"thalach":]
# creating boolean series

boolean = heart_data.chol > 400

heart_data[boolean]



#heart_data[heart_data.chol>400]
# combining filter

# hem chol değeri 200'den hem de thalach değeri 200'den büyük olan 

first_filter = heart_data.chol>200

second_filter = heart_data.thalach>200

heart_data[first_filter & second_filter]
# filtering column based others

heart_data.thalach[heart_data.chol>400]
# plain python function

def div(n):

    return n/2

heart_data.chol.apply(div)
# Or we can use lambda function

heart_data.chol.apply(lambda n: n/2)
# defining column using other columns

heart_data["total"] = heart_data.chol + heart_data.thalach

heart_data.head()
# index name

print(heart_data.index.name)

#lets change it

heart_data.index.name = "index_name"

heart_data.head()
# overwrite index

# if we want to modify index we need to change all of them

heart_data.head()
# first copy of our dta to data3 then change

data3 = heart_data.copy()



# lets make index start from 100.

data3.index = range(100,403,1)

data3.head()
# 

data1 = heart_data.set_index(["age","chol"])

data1 
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df = pd.DataFrame(dic)

df
# pivoting

df.pivot(index="treatment",columns = "gender",values="response")
heart_data.pivot_table(index="age", columns="sex", values="chol")