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
data.head()
data.info()
# Özellikler arasındaki correlatinı verir

data.corr()
# annot = true : kutuların içindeki sayıların gözükmesini sağlar

# linewidths : 2 kutunun arasındaki kalınlık

# fmt = 0 dan sonra yazılacak değer - 01 gibi



f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.columns
# datamızın columnlarından olan Speed'i al plot ettir 

# kind = line / scatter / histogram

# grid = True, grafik üzerindeki bölümlemeler



data.Speed.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# x = attack, y = defense

data.plot(kind='scatter', x='Attack', y='Defense',alpha = 0.5,color = 'red')

plt.xlabel('Attack')              # label = name of label

plt.ylabel('Defence')

plt.title('Attack Defense Scatter Plot')            # title = title of plot
# bins = barların sayısı

data.Speed.plot(kind = 'hist',bins = 50,figsize = (8,8), color="pink", grid=True)

plt.show()
data.Speed.plot(kind = 'hist',bins = 50)

plt.clf()
# dictionary oluşturma

dictionary = {'spain' : 'madrid' , 'usa' : 'vegas'}

print(dictionary.keys())

print(dictionary.values())
# dictionary elemanlarını update etme 

dictionary['spain']='barcelona'

print(dictionary)
# eleman ekleme

dictionary['france'] = 'paris'

print(dictionary)
# eleman silme



del dictionary['france']

print(dictionary)
#dictionary'de eleman arama



'spain' in dictionary
# dictionary silme

dictionary.clear()
print(dictionary)
# memoryde yer kalmasını istemiyorsak ;

del dictionary

print(dictionary) # olmadığı için hata verir
# Dosyayı okuma

data = pd.read_csv('/kaggle/input/pokemon-challenge/pokemon.csv')
series = data['Defense'] 

print(type(series))
dataframe = data[['Defense']]

print(type(dataframe))
x = data['Defense']>200

x.head(3)
data[x]
#logical_and = ikili filtelemede kullanılır

# numppy küt oldugu için np yazılır



data[np.logical_and(data['Defense']>200, data['Attack']>100)]



# 2.yol

# data[(data['Defense']>200) & (data['Attack']>100)]
data[np.logical_and(data['HP']<75, data['Sp. Def']>200) ]
i=0

while i!=5:

    print('i is:', i)

    i+=1

print(i, 'is equal to 5')
lst = [1,2,3,4,5]

for i in lst:

    print('i is:',i)

print('')
# listenin içindeki eleman ve indexlere erişim sağlamak istenildiğinde enumerate kullanılır

for index, value in enumerate(lst):

    print(index, " :",value)

print('')
# dictinary'de kullanmak için --> dictinary.items



dictinary = {'spain':'madrid','france':'paris'}



for key,value in dictinary.items():

    print(key, ' : ', value)

print('')
# dosyadan okunan veriler için for

# [0:1] = -> ilk elemandan 5. elemana kadar getir demek



for index, value in data[['Attack']][0:5].iterrows():

     print(index," : ",value)
def tuble_ex():

    """ t 'yi return eder'"""

    t = (14,25,36)

    return t



a,b,c = tuble_ex()

print(a,b,c)



# yap fonk. 2 değer döndürüyor a ve b. A yı kullanmak istiyoruz 

# b yi kullanmak istemiyoruz o zaman;

# a,_=yap()
x = 5 # global



def func():

    x=14 #local

    return x

print (x)

print (func())
# built in scope 'ları görmek için

import builtins

dir (builtins)
def square():

    """return aquares of value"""

    def add():

        """add 3 local variable"""

        x=5

        y=3

        z=7

        return x+y+z

    return add()**2

print(square())
def fun(a, b=1, c=2):

    y = a+b+c

    return y

print("*",fun(5))

print('')

print("*",fun(5,8,9))
def fun(*args):

    for i in args:

        print(i)

fun(1)

print("")

fun(2,4,6,8)

print("")

print("--")



#kwargs dictionary'de kullanılır



def fun(**kwargs):

    for key,value in kwargs.items():

        print(key, ': ', value)

fun( country= 'spain', capital='madrid', population= 123456)
square = lambda x:x**2

print(square(7))
summation = lambda x,y,z: x+y*z

print(summation(3,6,9))
num_list = {1,2,3}

y = map(lambda x:x**2, num_list)

print(list(y))
name = "ronaldo"

it = iter(name)

print(next(it)) #stringin ilk elemanını verir

print(*it) # geri kalanını verir
list1 = [1,2,3,4] #index olsun

list2 = [5,6,7,8] #indexe ailt elemanlar olsun

z = zip(list1,list2)

print(z)

print("-")

z_list = list(z)

print(z_list)
un_zip = zip(*z_list)

un_list1,un_list2 = list(un_zip) # unzip returns tuble

print(un_list1)

print(un_list2)

print(type(un_list2))



#print(type(list(unlist1))) listeye çevirme
num1=[1,2,3]

num2 = [i+1 for i in num1]

print(num2)
# conditionals on iterable



num1=[5,10,15]

num2=[i**2 if i==10 else i-5 if i<7 else i+5 for i in num1]

print(num2)
#pokemon csv dosyamızı data değişkenine atamıştık

data.head(1)
# csv dosyasındaki verimizde conditionals iterable yapma

threshold = sum (data.Speed)/len(data.Speed) #ort.hız

data["speed_level"]=["high" if i>threshold else "low" for i in data.Speed]

data.loc[:10,["speed_level","Speed"]]
data = pd.read_csv("/kaggle/input/pokemon-challenge/pokemon.csv")

data.head() #ilk 5 datayı gösterir
data.tail()#son 5 datayı gösterir
data.columns
data.shape 

# 800 row, 12 columns
data.info()
# value_counts()



d = data["Type 1"].value_counts(dropna=False)

print(d)
data.describe()
data.boxplot(column='Attack', by='Legendary')

plt.show()



#grafikte yuvarlak yerler aykırı değerleri göstermektedir
data_new = data.head()

data_new
# Data'da ismi değişmeden kalmasını sağlama : id_vars

# Yeni oluşması istenen datalar ise Attack ve Defense

melted = pd.melt(frame=data_new, id_vars="Name", value_vars=["Attack","Defense"])

melted
melted.pivot(index="Name", columns="variable", values="value")
#dikey concat

data1 = data.head()

data2 = data.tail()



conc_data_row = pd.concat([data1,data2], axis=0, ignore_index=True)

conc_data_row
#Yatay concat



data1 = data["Attack"].head()

data2 = data["Defense"].head()



concat2 = pd.concat([data1,data2], axis=1)

concat2
data.dtypes
# Convert etme



float("0.8")
data["Type 1"] = data["Type 1"].astype('category')

data["Speed"] = data["Speed"].astype('float')
data.dtypes
data.head(8)
data.info()
# Type 2'de kaç farklı değer var hesapla

# dropna=False -> Nan'larıda dahil et



data["Type 2"].value_counts(dropna=False)
# 386 missing value var

# Nan olanları listeden at

# inplace=True -> yapılan değişiklikleri kaydet



data["Type 2"].dropna(inplace=True)
#yukarıda yapılan işlemin işe yarayıp yaramadığını anlamak için;



assert 1 == 1



#birşey döndürmüyorsa doğru demektir
assert data["Type 2"].notnull().all()



#Type 2 deki hepsi notnull mı? 

## birşey döndürmediği için doğru


data["Type 2"].fillna('empty', inplace=True)
data.head(1)
assert data.columns[1] =="Name"



# 1.column Name old. için birşey döndürmez
assert data.Speed.dtypes == np.float
country = ["Turkey","England"]

population = ["100", "120"]



list_label= ["country","population"]

list_col = [country,population]



zipped = list(zip(list_label,list_col))



data_dict = dict(zipped)



df = pd.DataFrame(data_dict)

df
# yeni column ekleme



df["Capital"]=["Ankara","Londra"]

df
# Broadcasting : df yeni column ekleyip aynı değeri atama



df["Income"]=0

df
# Plotting Data



data1 = data.loc[:75,["Attack", "Defense", "Speed"]]

data1.plot()
# yukarıdaki grafiği farklı grafiklere bölmek için;



# Subplot



data1.plot(subplots=True)

plt.show()
# Scatter Plot

# corr bamak için kullanılır genelde

data1.plot(kind="scatter", x="Attack", y="Defense")
# Hist plot

# range = () : y eksenidir kaça kadar bakılması istenliyorsa o sayı yazılır

# normed = True : Veriyi 0-1 arasında normalize etmek demek



data1.plot(kind="hist", y="Speed", bins=50, color="pink", range=(0,120), normed=True)
# cumulative=True eklendiğinde 2. grafik oluşmaktadır

# cdf = cumulative distribution function

data1.plot(kind="hist", y="Speed", bins=50, color="orange", alpha=0.5, range=(0,120), normed=True)

data1.plot(kind="hist", y="Speed", bins=50, color="orange", range=(0,120), normed=True, cumulative=True)
# Statistical Exploratory Data Analysis



data.describe()
data.head()
# dataframe'e time columu ekleme



time_list = ["1992-03-08","1992-04-12"]

print(type(time_list))



dt_obj = pd.to_datetime(time_list)

print(type(dt_obj))
data2 = data.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

print("1. ",type(date_list))



datetime_object = pd.to_datetime(date_list)

print("2. ",type(datetime_object))



# columna ekleme

data2["date"] = datetime_object

data2 = data2.set_index("date")

# set_index = 1,2,3.., yerine date'i ekleme

data2
data2.info()
# indexler tarih oldu tarih seçerek bastırılır

print(data2.loc["1993-03-16"])
print(data2.loc["1992-03-10":"1993-03-16"])
# Yıla göre resampling yapıp ve bu verinin ortalamasını bulma

data2.resample("A").mean()
# yıllardaki ayları gösterir

data2.resample("M").mean().head(8)
# resample edilen datadaki NaN değerlerini doldurma

# interpolate("linear") : 1,2,3, ,5,6,7 oldug boşlupa 4 koyar



data2.resample("M").first().interpolate("linear").head(8)
data2.resample("M").mean().interpolate("linear").head(8)