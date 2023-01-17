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
data = pd.read_csv("../input/heart-disease-uci/heart.csv") # importing data
data.info() #şimdi de genel bir bilgi sahibi olalım
data.columns
data.head(10) #yukarıdan öğrendiklerimizle ilk 10 veriye bir göz atalım
#sırada bu verilerin birbirleri ile ilişkilerinin incelenmesi var

data.corr()
#bunu görselleştirelim - correlation map

f,ax = plt.subplots(figsize = (18,18))

sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt=".2f",ax=ax,cbar=True,linecolor="w")#cbar kenardaki göstergenin varlığını belirler

plt.show()
# line plot

# önce üzerinde çalışılan grubun yaş dağılımına bakalım

data.age.plot(kind="line",color="r",label="age",linewidth=2,alpha=1,grid=True,linestyle=":",figsize=(15,8))

plt.legend(loc=0)# loc=0 yerine 'best' de diyebilirdik. bu label ı en uygun yere koyar.

plt.xlabel("index")

plt.ylabel("age")

plt.title("yaş dağılımı")

plt.show()
#Histogram

data.age.plot(kind="hist",bins=25,figsize=(12,8))

plt.show()
# Açlık kan şekeri yüksek olan kişi sayısını bulalım

len(data[data.fbs == 1])
len(data[data.trestbps>120])
dataft = data[(data.trestbps > 120) & (data.fbs == 1)]

dataft.info()

dataft[dataft.target==1].info()
# scatter plot - trestbps ve chol arasındaki ilişkiye bakılması

# x=trestbps y=chol

data.plot(kind="scatter",x="trestbps",y="chol",color="red",alpha=0.5,figsize=(10,5))

plt.xlabel("trestbps")

plt.ylabel("chol")

plt.title("trestbps-chol ilişkisi")

plt.show()
data2 = data[data.target==1]

data2
#genel bir bilgi sahibi olalım

data2.describe()
# Hasta grupta egzersiz anjinasının görülme sıklığına bakalım şimdi de

data2.exang.plot(kind="hist",bins=10,color="c",figsize=(8,5))

plt.show()
data2.sex.plot(kind="hist",bins=10,color="m")

plt.show()
#sözlük oluşturma

sozluk = {"Türkiye":"TL","İngiltere":"sterlin"}

print(sozluk.keys())

print(sozluk.values())

print(sozluk.items())
sozluk["İngiltere"] = "ingiliz sterlini" # var olan entryi güncelledik

print(sozluk)

sozluk["USA"] = "USD" # yeni bir entry ekledik

print(sozluk)

del sozluk["İngiltere"] # entryi sildik

print(sozluk)

print("USA" in sozluk) # yazımından da anlaşılacağı gibi sözlükte bulunup bulunmadığını söyler

sozluk.clear() # sözlükteki bütün entryleri siler

print(sozluk)

# del sozluk komutu ise sözlüğü tamamen silecektir
# data = pd.read_csv("../input/heart-disease-uci/heart.csv") kodu ile datayı yukarıda import etmiştik
series = data['age']        

print(type(series))        # series-dataframe ilişkisi numpy daki vector-array ilişkisine benzer

dataframe = data[['age']] 

print(type(dataframe))

# thal = 2 (kalıcı defekt) olan kişi sayısını bulalım

liste_thal = []

for i in data.thal:

    if(i==2):

        liste_thal.append(i)

print(len(liste_thal))
# while döngüsü ile ufak bir sayı tahmin oyunu yapalım

deneme_hakki = 0 #sonraki blokların kesintiye uğramadan çalışması için deneme hakkını 0 yaptım. Siz çalıştırmak için 3 yapabilirsiniz.

xyz = np.random.randint(0,5)

while deneme_hakki>0:

    sayi = int(input("Bir sayı giriniz: ")) # input fonksiyonu ile aldığımız değer string olur,buna dikkat etmezsek kodumuz çalışmayacaktır.

    if (xyz==sayi):

        print("Tebrikler, doğru bildiniz. Sayımız: {} ".format(sayi))

        break

    else:

        deneme_hakki -= 1

        print("yanlış tahmin")

    if deneme_hakki == 0:

        print("denem hakkınız tükendi")
# enumerate fonksiyonu

lis = ["a","b","c","d","e"]

for index,value in enumerate(lis):

    print(index," : ",value)

print("\n") # \n bir alt satıra geçmeyi sağlar.print fonksiyonunda bu zaten vardır,her seferinde bir alt satıra geçer.o yüzden burada iki alt satıra geçti.

# sözluklerde items metodu

sozluk = {"a":"x","b":"y"}

for key,value in sozluk.items():

    print(key," : ",value)

print("\n")

# iterrows

for index,value in data[["cp"]][0:3].iterrows():

    print(index," : ",value)
for i,j in data.iterrows():   # burada i satır indeksini, j ise satırı ifade eder. İlk satırı alalım

    print("i: ",i," j: ",j)

    break
# thalach değeri ortalama değerden fazla olanlarla bir dataframe oluşturalım

data_thalach = pd.DataFrame()

for i,j in data.iterrows():

    if(j.thalach>data.thalach.mean()):

        data_thalach = data_thalach.append(j)

data_thalach.info()
#son olarak da yeni dataframimizi görselleştirelim

data_thalach.thalach.plot(kind="line",color="m",label="thalach",linewidth=1,alpha=0.9,grid=True,linestyle="--",figsize=(12,8))

plt.legend(loc=0)

plt.xlabel("index")

plt.ylabel("thalach")

plt.title("thalach değeri ortalamadan büyük olanlar")

plt.show()