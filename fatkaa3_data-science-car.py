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
data = pd.read_csv('../input/car_ad.csv',encoding ='iso-8859-9')
data.shape # kac satır kaç sütun var
data.columns # sütun isimleri
data.info() # dataset hakkında genel bilgiler 
data.isnull().sum() # eksik verileri kontrol ediyoruz
data.describe() # datasetdeki sayısal değerler hakkında bilgiler
# ör. arabaların ortalama kilometresi, max kilometre, min kilometre 
data.head() # ilk 5 veri
# sütunlarda ne kadar farklı değer var 
print(len(data.car.unique()))
data.car.unique() # (ör. kaç farklı araba markası var)
#data.engType.unique()
data.year.unique()
#data.model.unique()
#data.drive.unique()
data.corr() 
# Scatter plot ile yıl-fiyat kolerasyonu 
data.plot(kind = 'scatter', x='year', y = 'price', alpha = 0.5, color = 'r')
plt.xlabel("year")
plt.ylabel("price")
plt.title("year - price Scatter Plot")
plt.show()
data_vc = data['car'].value_counts() # araba isimlerine göre grupluyor 
data_vc.head()
volkswagen = data[data.car == 'Volkswagen'] # Volkswagen olan arabaları ayrıyoruz
volkswagen.head(10) # 10 tanesini yazdırıyoruz

mercedes_benz = data[data.car == 'Mercedes-Benz']# Mercedes-Benz olan arabaları ayrıyoruz
mercedes_benz.head(10) # 10 tanesini yazdırıyoruz
#Volkswagen yıllara göre sayılarını görselleştiriyoruz 
plt.hist(volkswagen.year, bins = 30)
plt.xlabel("year")
plt.ylabel("frekans")
plt.title("Volkswagen Hist")
plt.legend()
plt.show()
#Mercedes-Benz yıllara göre sayılarını görselleştiriyoruz 
plt.hist(mercedes_benz.year, bins = 30)
plt.xlabel("year")
plt.ylabel("frekans")
plt.title("Mercedes-Benz Hist")
plt.legend()
plt.show()
#yılı 2012 den büyük olanlar
x = data['year'] > 2012 
#x 
data[x]
# yılı 2012 den , motor boyutu da 3,00 den büyük olanlar
#data[(data['year'] > 2012) & (data['engV'] > 3.00)] # bu şekilde de yapılabilir 
data[np.logical_and(data['year'] > 2012 , data['engV'] > 3.00)]
#kendi isteğimize özel filtreleme yapıp ilk veriyi yazıdırdık
data_ozel = data[np.logical_and(data['car'] == 'Ford' , data['model'] == 'Mustang')]
for index, value in data_ozel[0:1].iterrows():
    print(index, ' : ', value)
engV = 5.0 # global
#user defined fonction
def Bmw(model, engV, car = 'BMW'):
    """BMW marka arabaların istenilen model ve motor hacmine göre listelenmesi"""
    data_ozel = data[(data['car'] == car) & ( data['model'] == model) & (data['engV'] == engV)]
    return data_ozel
data_ozel =Bmw('M5',engV)
data_ozel
ort = sum(data['mileage']) / len(data['mileage'])
print("ortalama kilometre: ", ort)
data['mileage_level'] = ["high mileage" if i > ort else "low mileage" for i in data['mileage']]
data.loc[:10]
data['year_model_level'] = ["eski model" if i<2010 else "yeni model" if(2016 > i) else "son model" for i in data['year']]
data.head(20)

data['kapi_sayisi'] = 2
data
data['kapi_sayisi'] =list(map(lambda  x: x + 2, data['kapi_sayisi']))
data