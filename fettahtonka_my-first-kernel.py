# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# 1. Tür Ele Alış Biçimi
Kütüphane={"Ülkeler":["Türkiye","İspanya","Almanya","İngiltere","Fransa"],
          "Yaş_Ortalaması":["31,7","43,6","46,0","40,1","41,6"],
          "Nüfus(Milyon)": ["84.400.000","47.100.396","83.149.300","67.886.011","67.075.000"]}
Veri=pd.DataFrame(Kütüphane)
Veri.columns #Sütun İsimlerini Verir
#2.Tür Ele Alış Biçimi
Data=pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")
DT=pd.read_csv("../input/videogamesales/vgsales.csv")
DT.columns
DT.dtypes
#Sorgulama İşlemleri
Data.shape #Satır,Sütun Sayısını Verir
Data.columns #Sütun İsimlerini Verir
Data.dtypes #Sütunların Veri Tiplerini Verir
Data.head #İlk 5 Elemanı Verir
# Data.head(X) İlk X Tane Eleman Verir
Data.tail() #Son 5 Elemanı Verir
# Data.tail(X) Son X Elemanı Verir
Data.info #Dosyalar Hakkında Bazı Bilgileri Barındırır
Data.describe() #Bazı İstatisel Bilgileri Verir
Data.count() #Nan olmayan Girdi Sayısını Verir
#Filtreleme
Data.columns
Data["Rating"]>4.0 # True/False Olarak Çevirir
Data[Data["Rating"]>4.0] # Özelliği Sağlayanları Satır Bilgileri ile çevirir
Data[Data["Category"]=="FAMILY"] #Özelliği Saplayanları Saır Bilgileri İle Çevirir
Data.loc[2,:] #2.Satır Tüm Sütunlar
Data.loc[0:2,:] #Sıdırdan 2.Satıra Ve Tüm Sütunlar
Data.loc[:,"Rating"] #Tüm Satırlar Sadece "Rating " Sütunu
Data.loc[:,["Rating","Reviews"]] #Tüm Satırlar Ve Sadece Rating ve Reviews Sütunları
Data[(Data["Category"]=="FAMILY")&(Data["Rating"]>4.5)]
#Data["Yeni_Özk"]=["İyi İzlenme" if i>4.2 else "Kötü İzlenme" for i in Data.Rating] #Filtreleme Yöntemi İle Yeni Özk Ekleme
#2 Listeyi Birleştirme
DATA1=Data.head()
DATA2=Data.tail()
Data1=pd.concat([DATA1,DATA2],axis=0) #Listelerin Yatay Olarak Birleşmesini Sağlar
Data2=pd.concat([DATA1,DATA2],axis=1) #Listelerin Dikey Olarak Birlşemesini Sağlar
#Sözlük İle Çalışmak
Dictionary={"Almanya":"Audi","Amerika Birleşik Devletleri":"Ford","İngiltere":"Range_Rover","Japan":"Toyota"}
Dictionary.keys() # Sözlükteki Anahtar Kelimeleri Dökme
Dictionary.values() #Sözlükteki Değerleri Dökme
Dictionary["Almanya"]="Mercedes-Benz" # Değer Değiştirme
del Dictionary["Japan"] # Sözlükten Eleman Silme
"Almanya" in Dictionary # Almanya Sözlükte Varmı ? True Yada False
# Dictionary.clear() == Sözlüğü Temizler
# del Dictionary == Sözlüğü Tamamen Silme
#For-While Loop

#While-Loop
i=0
while i!=10:
 print("Lim",i)
 i+=1
print("limon Oldu ...")

#For-Loop
Dictionary={"Türkiye :":"İstanbul ","Almanya :":"Düsseldorf","USA :":"San Francisco"}
for key,value in Dictionary.items():
    print(key,"",value)

#Matplotlib
DT.Year.plot(kind="line",color="green",linewidth="1",alpha=0.5,grid=True,label="Global_Sales")
DT.Global_Sales.plot(color="blue",linewidth="1",alpha=0.7,linestyle="-",label="Year")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Global_Sales")
plt.show()
DT.plot(kind="scatter",x="JP_Sales",y="EU_Sales",color="blue")
plt.xlabel("JP_Sales")
plt.ylabel("EU_Sales")
plt.title("JP_Sales/EU_Sales")
plt.show()