#DATAI TEAM 
#Kütüphaneleri ekle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Oynamak istediğin datayı import et
data = pd.read_csv('../input/pokemon.csv')
#Data hakkında detaylı bilgi al
data.info()
#Datanın değerleri hakkında grafiksel bilgi al
data.corr()
#Datanın değerlerini sutünlar halinde görselleştir.
#Datanın tablosu 18,18 olsun
#Datanın içindeki sayılar gözüksün,grafikte sutünlar arası kalınlık .5 olsun, 0'dan sonra 1 sayı gözüksün
f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(),annot=True,linewidth=.5,fmt='.1f',x=ax)
plt.show()
#Datadaki ilk 5 değeri göster
data.head()
#Datanın sahip olduğu  kolon isimlerini yazdır
data.columns
#Lane Plot
#Hız kolonunu plot ettir--Türü:Line,Rengi:Yeşil,Etiketi:Hız,Çizgi Kallınlığı:1,Şeffaflık:0.5,Kafes=Doğru,Çizgi Stili=:
data.Speed.plot(kind='line',color='g',label='Speed',linewidth=1,alpha=.5,grid=True,linestyle=':')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Graphic')
plt.show()
#Scatter Plot
#Datayı plot ettir--Türü=Scatter,X axis=Attack,Y axis=Defense,Şeffaflık 0.5,Renk:Kırmızı
data.plot(kind='scatter',x='Attack',y='Defense',alpha=0.5,color='red')
plt.xlabel('Attack')
plt.ylabel('Defense')
plt.title('Scatter Graphic')
plt.show()
#Histogram Plot
#Datanın hızını plot ettir--Tür:Hist,50 tane kolon'cizgi' olsun,büyüklüğü 12,12 olsun.
data.Speed.plot(kind='hist',bins=50,figsize=(12,12))
plt.show()
#Bir histogram plotu oluştur ve onu temizle
data.Speed.plot(kind='hist',bins=50)
plt.clf()


