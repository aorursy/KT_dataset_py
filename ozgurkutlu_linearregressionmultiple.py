# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# csv dosyası data'ya import ediliyor

data = np.genfromtxt('../input/evfiyatlari3feature.csv', delimiter=',', skip_header=True)

# gerekli kütüphaneler import ediliyor

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error as MAE

from sklearn.metrics import mean_squared_error as MSE

from sklearn.model_selection import cross_val_score as CVS

from sklearn.preprocessing import OneHotEncoder
# data içerisindeki sütunlar ilgili numpy dizilere atanıyor

ev_m2 = data[:, 0]

ev_kat = data[:, 1]

ev_renk = data[:, 2]

ev_fiyat = data[:, 3]

# Grafikler çizdiriliyor. Grafiklerden anlaşılan Kat ve renk bilgileri 

#katogorik bilgiler(sayısal değerlere sahip olsalarda) 

#bu yüzden direk bu sütunlarla fit etmek yanlış sonuç almamıza sebeb olur



fig=plt.figure(figsize=(15,10))

ax1 = fig.add_subplot(2, 2, 1)

ax2 = fig.add_subplot(2, 2, 2)

ax3 = fig.add_subplot(2, 2, 3)



ax1.scatter(ev_m2,ev_fiyat)

ax1.set_title('Metrekare-Fiyat')

ax1.set_xlabel('Metrekare')

ax1.set_ylabel('Fiyat')



ax2.scatter(ev_kat,ev_fiyat)

ax2.set_title('Kat-Fiyat')

ax2.set_xlabel('Kat')

ax2.set_ylabel('Fiyat')



ax3.scatter(ev_renk,ev_fiyat)

ax3.set_title('Renk-Fiyat')

ax3.set_xlabel('Renk')

ax3.set_ylabel('Fiyat')

plt.show()
#sklearn kütüphanesinin istediği boyutlandırma işlemleri yapılıp tek feature göre fit ediliyor.Referans olabilmesi için

ev_m2=ev_m2.reshape(-1,1)

ev_kat=ev_kat.reshape(-1,1)

ev_renk=ev_renk.reshape(-1,1)

ev_fiyat=ev_fiyat.reshape(-1,1)

regresyon = LinearRegression()

regresyon.fit( ev_m2, ev_fiyat )

# eğim-sabit, mse ve r2(score) değerleri yazdırılıyor.(r2 1 e ne kadar yakınsa okadar iyi fit etmiş demektir)

print('sklearn m : ', regresyon.coef_)

print('sklearn b : ', regresyon.intercept_)

MSE_CVS_degerleri = - CVS( regresyon, ev_m2, ev_fiyat, cv = 5, scoring='neg_mean_squared_error' )

print('MSE_CVS_degerleri:',MSE_CVS_degerleri)

print('score:',regresyon.score(ev_m2, ev_fiyat))
#fit ettiğimiz line çizdiriliyor

ev_fiyat_2B_tahmin = regresyon.predict(ev_m2)

plt.scatter(ev_m2, ev_fiyat_2B_tahmin)

plt.xlabel("metrekare")

plt.ylabel("fiyat")

plt.show()
# multiple regresyon için 2 sütunu içeren ev_mult arrayı oluşturuluyor ama hala kat ve renk bilgisinin 

# katagorik olduğunu dikkate almadık. Bakalım bu şekilde score yani r2 ne çıkacak

ev_mult = data[:, 0:3] #renk sütunu  dahil 

regresyon2 = LinearRegression()

regresyon2.fit( ev_mult, ev_fiyat )
print('score:',regresyon2.score(ev_mult, ev_fiyat))
# ev_kat ve ev_renk sütunundaki bilgiler onehotencoder işlemine tabi tutuluyor böylece numpynin

# işleyeceği 000010000000 gibi 12 sütun 100 satırlık bir array haline geliyor(Her katagori bir sütunda temsil ediliyor)

ohe = OneHotEncoder(sparse=False)

ev_kat_ohe=ohe.fit_transform(ev_kat)

ev_renk_ohe=ohe.fit_transform(ev_renk)
#elde edilen ev_ohe ile ev_m2 arrayleri birleştirilerek fit işlemine beraberce veriliyor

birlesik=np.concatenate((ev_m2,ev_kat_ohe,ev_renk_ohe),axis=1)

regresyon3 = LinearRegression()

regresyon3.fit( birlesik, ev_fiyat )
# score(r2) değerine bakılınca şuana kadarki en yüksek değer olduğu görülüyor

print('score:',regresyon3.score(birlesik, ev_fiyat))