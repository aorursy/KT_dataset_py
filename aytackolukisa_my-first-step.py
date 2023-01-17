# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_prices = pd.read_csv("/kaggle/input/nyse/prices.csv")

data_split = pd.read_csv("/kaggle/input/nyse/prices-split-adjusted.csv")

data_fundamentals = pd.read_csv("/kaggle/input/nyse/fundamentals.csv")

data_securiries = pd.read_csv("/kaggle/input/nyse/securities.csv")
import matplotlib.pyplot as plt

import seaborn as sns
data_prices.info()#.info() data frame hakkında bizi gemel bilgiler veren bir fonksiyondur
data_prices.columns#kısaca bir columns lara göz atalım
data_prices.head(20)# dataframemizin içinde yer alan ilk 20 data ya bakalım
#correlation map

f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(data_prices.corr(),annot=True, linewidths=.5, fmt=".1f", ax=ax)
plt.plot(data_prices.date,data_prices.close,color = 'red',label = 'Close/Date',alpha = 0.5,)

plt.xlabel=("x axis")

plt.ylabel=("y axis")

plt.legend(loc='upper left') 

plt.show()
plt.hist(data_prices.close,bins=10 )

plt.xlabel("close price")

plt.ylabel("frekans")

plt.title("hist of close price")

plt.show()
plt.scatter(data_prices.volume,data_prices.close,color = 'red',label="Volume/Close_Price")

plt.xlabel("volume")

plt.ylabel("close price")

plt.title("scatter of close price")

plt.show()
seri_low=data_prices["low"] #data_price içinde yer alan low sütününu seri dizgisi haline çevirdik.

data_low=data_prices[["low"]] #data_pice içinde yer alan low sütününu dateframe e çevirdik dizgisi haline çevirdik.

print(type(seri_low))

print(type(data_low))
dic={"Ali":12,

     "Ayse":21,

     "Mehmet":23}



print(dic)

print(dic.keys())

print(dic.values())
#Yeni key ve value ekleyelim

dic["Fırat"]="31"

print(dic)
#key sabit kalsuın value değişikliği yapalım

dic["Ali"]="33"

print(dic)
#Keylerden birtanesini remove edelim

del dic["Mehmet"]

print(dic)
#checking

if "Ayse"in dic:

    print("Ayse dicin içinde")

else:

    print("Aradığınız Ayse ulaşılamamktadır")

#veya

print("Ayse"in dic)
#dictionary içindeki tüm bilgileri silmek istersek

#dic.clear()

print(dic)
#Dicitioanary i ramdan de silmek istersek

#del dic

#print(dic)
#Filtreleme

x=data_prices["open"]>1500.000

data_prices[x]
data_prices[np.logical_and(data_prices["close"]>1500.00,data_prices["volume"]>327500)]

#numpy kütüphanesi yardımı ile yapıla bir kod yazılımıdır.
data_prices[(data_prices["close"]>1500) & (data_prices["volume"]>327500)]

# Aynı sonucu farklı kütüphaneler kullanarakta alabilyoruz
x=0

while x!=6:

    x+=1

    print("x=",x)

print(x,"esittir 6")
list1=[1,2,3,4,5,6,7]

for x in list1:

    print(x,"eşitir x")
#Listede yeralan elemanların indexleri ni öğrenmek için

for index,value in enumerate(list1):

    print(index,"",value)
#Dictionary içinde yer alan keys ve value değerlerine sırası ile ulaşmak için for döngüsü kullanılır

for key,value in dic.items():

    print(key,"",value)
# Bir data frame içimdeki bilgilerede ulaşmak için for döngüsü kullanılır

for index, value in data_prices[["close"]][0:1].iterrows():# data_price içinde close sütünunun ilk elemanının index ve value sine eriştik

    print(index,"",value)
# Kısaca özetlemek gerekirse fonksitonun içinde tanımlanan scope Global scope içinde tanımlanan ise local scope

x=3

def f():

    x=4

    return x

print(x)

print(f())
#Eğer fonksiyonun içinde x tanımlı depise global x i alır.

x=3

def f():

    y=x**2

    return y

print(f())
def square ():

    def add():

        x=2

        y=3

        z=y*x

        return z

    return add()**2

print (square())

def f (a, b=1, c=2):

    x=a*b+c

    return x

f(3)
#Eğer biz önceden atadığımız b ve c değerlerini değiştirecek olursak 

f(3,2,3)
def f(x,*arg):

    x=2

    y=3

    z=4

    n=x*y*z

    return n

print( f(x))

#Yukarıda görüleceği üzer fonksiyonu yazarken sadece x i tanımladık diğer values leri ise

#fonksiyonun içinde belirledik bunun için *arg kulanadık bu sayede fonksiyonun içinde sonsuz değişken tanımlayabiliriz.
def f(**kwargs) : # dictionary içine kullanılan ifade

    for key,value in kwargs.items():

        print(key,":",value)

f(Team="Galatasaray",Mannager="Fatih Terim",Fans="35000000")
square=lambda x: x**3

print(square(2))
tot=lambda x,y,z:x+y+z

print(tot(1,2,3))
list_2=[1,2,3,4]

y=map(lambda x:x**2,list_2)

print(list(y))
list4=[1,2,3,4]

list5=[21,32,43,54]

z=zip(list4,list5)

print(z)

z_list=list(z)

print(z_list)
unzip=zip(*z_list)

print(unzip)# zip un zip yapıldı ama şuan ram içeresinde

unlist1,unlist2=list(unzip)#Şimdide unzip i iki farkli listeye dönderdik

print(unlist1)# Görüleceği gibi listelre tuple şeklinde

print(unlist2)

a_list=list(unlist1)

print(type(a_list))

print(a_list)# evet listemiz köşeli hale geldi
num1=[1,2,3]

num2=[x+1 for x in num1]

print(num2)
num=[12,15,23]

num1=[x+5 if x==15 else x*2 if x<13 else x-2 for x in num]

print(num1)
nn=[7,16,32,44]

nn1=[x+2 if x==16 else x*2 if x<13 else x-5 if 24<x<35 else x/4 for x in nn]

print(nn1)
avrg_price=sum(data_prices.close)/len(data_prices)

print(str(avrg_price))
data_prices["price_condition"]=["Price is high" if x > str(avrg_price) else "Price is low" for x in data_prices]

data_prices.loc[:10,["close","price_condition"]]
data_prices = pd.read_csv("/kaggle/input/nyse/prices.csv")
data_new=data_prices.head()
data_new
data_prices.tail()
melted = pd.melt(frame = data_new,id_vars = "symbol", value_vars=("close","open"))

melted

data_new2=data_prices.tail()

data_conc = pd.concat([data_new,data_new2],axis=0,ignore_index=True)

data_conc
data_conc_yatay = pd.concat([data_new,data_new2],axis=1,ignore_index=True)

data_conc_yatay
data_new2
data_new2.dtypes
data_new2["open"]=data_new2["open"].astype("int64")
name=["aytac","ebru","zeycan","arda","canan","mahmut","mehmet","hatice","berkay","azzet","elif","koray","mert","senay"]

age=[21,23,17,5,40,44,6,70,23,33,16,43,45,88]

list_label=["name","age"]

list_columns=[name,age]

zipped=list(zip(list_label,list_columns))

data_dic=dict(zipped)

dataf=pd.DataFrame(data_dic)

dataf

dataf["sex"]=["m","f","f","m","f","m","m","f","m","f","f","m","m","f"]

dataf["bugget"]=[500,600,400,700,50,640,1300,2500,800,750,30,2100,1100,90,]

dataf
import matplotlib.pyplot as plt
dataf.boxplot(column="bugget",by="sex")

plt.show()
data1=dataf.loc[:,["age","bugget"]]

data1.plot()

plt.show()
data1.plot(subplots=True)

plt.show()
data1.plot(kind="scatter",x="age",y="bugget")

plt.show()
time_series=["2018-01-01","2018-02-01","2018-03-01","2018-07-01","2018-08-01","2018-09-01","2018-10-01","2018-11-01","2018-12-01","2019-01-01","2019-04-01","2019-05-01","2019-06-01","2019-07-01"]
date_time=pd.to_datetime(time_series)

type(time_series)

type(date_time)
dataf["date"]=date_time

dataf=dataf.set_index("date")

dataf
print(dataf.loc["2018-01-01"])
print(dataf.loc["2018-01-01":"2018-11-01"])
dataf.resample("A").mean()
dataf.resample("M").mean()
dataf.resample("M").first().interpolate("linear")
dataf.resample("M").mean().interpolate("linear")
dataz=data_prices.head(10)
dataz
dataz["nember"]=[1,2,3,4,5,6,7,8,9,10]

dataz
dataz=dataz.set_index('nember')

dataz
dataz["date"][10]
dataz[["date","close"]]
dataz.loc[1:5,"date":"low"]
data_prices=pd.read_csv("/kaggle/input/nyse/prices.csv")
datax=data_prices.head(150)
datax
ff=datax.close>120

sf=datax.volume>2100000

datax[ff&sf]
datax.open[datax.low>124]
def div (n):

   return n/2



print(div(2))
datax.volume.apply(div)
datax.open.apply(lambda n: n/2)
datax["mum_boyu"]=datax.close-datax.open

datax
data_prices=data_prices.set_index(["open","close"])

data_prices.head()
dic={"Team":["Galatasaray","Fenerbahçe","Besiktas","Trabzon"],

    "lig_Cup":[22,19,16,9],

    "Bugget":[100000,75000,67000,35000],

    "Fans":[35000,23000,15000,5000],

    "Number":["1","2","3","4"]}

df=pd.DataFrame(dic)
df
df = df.set_index('Number')

df
df.pivot(index="Team",columns="Bugget",values="lig_Cup")
df.groupby("lig_Cup").mean()