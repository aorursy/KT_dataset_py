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
#correlation map
f,ax = plt.subplots(figsize=(18,18)) # figsize şeklin boyutunu belirtiyor.
sns.heatmap(data.corr(),annot=True,linewidths=.5, fmt=".1f",ax=ax) # data.corr() ilişkisel bir excel tablosu döndürür.
#attribute ler arasında doğru orantı veya ters orantılı bir ilişki varmı onu verir
#  0 ise ilişki yoktur
#  1 ise ilişki vardır
#plt.show() yazınca çıktı verilmez.
plt.show()
data.head(10)
#ilk 10 satırı ekrana getirir.
data.columns 
# sahip olunan attributeları verir.
#Matplot is a python library that help us to plot data. 
#The easiest and basic plots are line, scatter and histogram plots.

#scatter iki değer arasında correlation varmı diye bakarken
#histogram bir data nın dağılımına(sıklığına) bakarken kullanılır.
#line x axis i zaman ise kullanmak güzeldir.

#Line plot örneği
data.Speed.plot(kind="line",color="g",label="Speed",linewidth=1,alpha=0.5,grid=True,linestyle=":")
data.Defense.plot(kind="line",color="r",label="Defense",linewidth=1,alpha=0.5,grid=True,linestyle="-.")
plt.legend(loc="upper right")
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()
#Scatter plat örneği
#Attack ile defense arasındaki ilişki ne onu bulmak amaçlı
data.plot(kind="scatter",x="Attack",y="Defense",alpha=0.5,color="red",grid=True)

#plt.scatter(data.Attack,data.Defense,alpha=0.5,color="red")

plt.xlabel("Attack")
plt.ylabel("Defence")
plt.title("Attack Defence Scatter Plot")
plt.show()

#Histogram Plot örneği
#x ekseni pokemonların hız değerleri
#y ekseni frekansları
#bins = 50 demek 50 adet bar olsun demektir.
data.Speed.plot(kind="hist",bins=50,figsize = (20,10))
plt.xlabel("Speed of Pokemon")
plt.show()
dictionary = { "spain":"madrid","usa":"vegas"}
print(dictionary.keys())
print(dictionary.values())
dictionary["spain"] = "barcelona"
print(dictionary)
del dictionary["spain"]
print(dictionary)
print("usa" in dictionary)
#dictionary.clear()
#print(dictionary)
#del dictionary
#print(dictionary)
#Dosya import etme CSV formatındaki dosyalar
data = pd.read_csv("/kaggle/input/pokemon-challenge/pokemon.csv")
data
Series = data["Defense"]  # data['Defense'] = series
print(Series,"\n")
data_frame = data[["Defense"]] # data[ ['Defense'] ] = data frame
print(data_frame)
x = data['Defense'] > 200  # data['Defense'] > 200 demek bir boolean seri döndürür.
data[x]
#Birden fazla koşul için
x = (data['Defense'] > 200) & (data['Attack']>100)
data[x]
#Birden fazla koşul için numpy kütüphanesi ve logical_and kullanarak ( logical_or )
data[ np.logical_and( data['Defense']>200 , data['Attack']>100 ) ]
lis = [1,2,3,4,5]

for i in lis:
    print("i: ",i)
print()

#Enumerate kullanarak
for index,value in enumerate(lis):
    print(index,".indexteki value : ",value)
print()

#dictionary ler için
for key,value in dictionary.items():
    print(key," : ",value)
print()

#pandas için
for index,value in data[['Attack']][1:2].iterrows(): # 1.satırdan 2.satıra kadar olanlarn Attack sütununu getir..
    print(index," : ",value)
#Nested Function

def five_exponent(parameter):
    """return five exponent parameter"""
    def add():
        """add two local variable"""
        x= 2
        y= 3
        z= x+y
        return z
    return add()**parameter
five_exponent(2)
#Flexible Arguments
def f(*args): #listede kullanılıyor tek yıldız
    for i in args:
        print(i)
def g(**args): #dict de kullanılıyor çift yıldız
    for key,value in args.items():
        print(key," : ",value)
f(1)
f(1,2,3,4)
g(country="spain",capital ="madrid")
#Lamda Function
square = lambda x: x**2
print(square(4))
total = lambda x,y,z : x+y+z
print(total(1,2,3))
#Anonymous Function ( Map(function,parameter) )
number_list = [1,2,3]
number_list = map(lambda x:x**2,number_list)
print(list(number_list))

#Iterators 
#for while ı zaten iterator olarak kullanıyoruz

liste = [1,2,3]
iterator = iter(liste)
print(next(iterator))
print(*iterator)

#zip function example
list1 = ["338231234","12312312"]
list2 = ["Muhamemd Said","Ahmet Kerem"]

insanlar = list( zip(list1,list2) )
print(insanlar)
print(type(insanlar))
print(type(insanlar[0]))

#unzip function example
un_zip = zip(*insanlar)
list1 , list2 = list(un_zip)
list1 = list(list1)
list2 = list(list2)
print(list1)
print(list2)


#List Comprehension
#It is realy important
num1 = [1,2,3]
num2 = [i+1 for i in num1] #example1
print(num2)

num1 = [5,10,15]
num2 = [ i**2 if i==10 else i-5 if i<7 else i+5 for i in num1] #example2
print(num2)


#Pandas list comprehension

threshold = sum(data.Speed)/len(data.Speed) #ortalama hız bulma

data['speed_level'] = [ "high" if i > threshold else "low" for i in data.Speed]

data.loc[:10,["speed_level","Speed"]] # ilk 10 satırı getir .
print(data["speed_level"].value_counts(dropna=False))
data.describe()
data.boxplot(column="Attack",by = "Legendary")
plt.show() # Yuvarlak lar outlier dır.
data_new = data.head()
data_new
#melt fonksiyonu farklı bir yapıya dönüştürüyor. Yapılma nedeni melt fonksiyonunu kullanarak seaborn
#kütüphanesinde daha güzel gösterebilmektir. Bir nevi atomic yapıya dönüştürüyor
melted = pd.melt(frame=data_new,id_vars="Name",value_vars=["Attack","Defense"])
melted
melted.pivot(index="Name",columns="variable",values="value")
# Vertical
data1 = data.head()
data2 = data.tail()
conc_data_row = pd.concat([data1,data2],axis=0,ignore_index=True)
conc_data_row
# Horizontal
data3 = data.loc[10:20,"Name"]
data4 = data.loc[10:20,"Attack"]
conc_data_col = pd.concat([data3,data4],axis=1,ignore_index=False)
conc_data_col
data.dtypes
# Bütün sütünun type ını değiştirme
data["Type 1"] = data["Type 1"].astype("category")
data["Speed"] = data["Speed"].astype("float")
data.dtypes
#Groupby example
data.groupby("Type 1")["Attack"].mean()
data.info()
data["Type 2"].value_counts(dropna=False) # 386 adet Nan var
data["Type 2"].dropna(inplace=True)   # So does it work ? 
assert data["Type 2"].notnull().all() # returns nothing because we drop nan values
data["Type 2"].value_counts(dropna=False) # 0 adet Nan var
data["Type 2"].fillna("empty",inplace=True)
assert data["Type 2"].notnull().all() # returns nothing because we do not have nan values
data.head(10)
country = ["Spain","France"]
population = ["12","11"]
list_label = ["country","population"]
list_col = [country,population]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df
# Add new column
df["capital"] = ["madrid","paris"]
# Broadcasting
df["income"] = 0 # Broadcasting entire column
df
import warnings
warnings.filterwarnings("ignore")

time_list = ["1992-03-08","1992-04-12"]
datetime_object = pd.to_datetime(time_list)

data2 = data.head(2)
data2["date"] = datetime_object
data2 = data2.set_index("date")
data2.loc["1992-03-01":"1992-05-15"]
data2.resample("A").mean()   # yıllara göre değerlerin ortalamasını alma
data2.resample("M").mean()   # aylara göre değerlerin ortancasını alma
data = pd.read_csv("../input/pokemon-challenge/pokemon.csv")
data.set_index("#",inplace=True)
data
# Indexing using square brackets
data["HP"][1]
# using column attribute and row label
data.HP[1]
# using loc accessor
data.loc[1,"HP"]
# selecting only some columns
data[["HP","Attack"]] + data.loc[:,["HP","Attack"]] # + veya anlamında iki şekildede olabilir.
boolean = data.HP > 200
data[boolean]
# Combining filters
data[ (data.HP > 200) & (data.Speed > 35)]
# Filtering column based others
data.HP[data.Speed>15] # Hızı 15 den küçük olanları HP lerini almak amacıyla
data.HP.apply(lambda x: x*2)
print(data.index.name)
data.index.name = "primary key"  # index name değiştirme
data.head()
data.index =range(100,900,1)  # index in range ini değiştirme
data.head()
data.set_index("Name")  # bir column u index yapma
data = pd.read_csv("../input/pokemon-challenge/pokemon.csv")
data.head()
data.set_index(["Type 1","Type 2"],inplace=True) # Type 1 outer index Type 2 inner index
data.head(100)
dic = { 
    "treatment":["A","A","B","B"],
    "gender":["F","M","F","M"],
    "response":[10,45,5,9],
    "age":[15,4,72,65]
      }
df = pd.DataFrame(dic)
df
df.pivot(index="treatment",columns="gender",values="response")
df1 = df.set_index(["treatment","gender"])
df1
#unstack
df1.unstack(level=1)
df1.swaplevel(0,1) # indexlerin yerini değiştirme
# Melt ->Pivot un tersi
pd.melt(df,id_vars="treatment",value_vars=["age","response"])
df
df.groupby("treatment").mean()
df.groupby("treatment")["age"].max()
df.groupby("treatment")[["age","response"]].max()
a = [0,1,2,3,4]

for a[0] in a:

    print(a[0])
