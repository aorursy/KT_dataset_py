# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_tests = pd.read_csv('../input/covid19-in-turkey/test_numbers.csv')

data = pd.read_csv('../input/covid19-in-turkey/covid_19_data_tr.csv')
data.info()
#DataFrame tablosunda sütun isimlerini türkçeleri ile değiştirdik.

#Değer girilmemiş "Province/State" sütununu DataFrameden çıkardık.

data.rename(columns = {"Country/Region":"Ülke","Last_Update" : "Tarih", "Confirmed" : "Vaka_Sayisi", "Deaths" : "Vefat_Sayisi", "Recovered" : "Tedavi_Sayisi"}, inplace = True)

data.drop("Province/State", axis = 1, inplace = True)

data.info()
data_tests
#TABLO DÜZENLEMELERİ

#Test sayısı bilgilerini "data" dataframe'ine yeni sütun olarak ekledik. 

test_sayisi = data_tests.iloc[0,4::].values

test_sayisi.sort()

data['Test_Sayisi'] = test_sayisi 

data.columns
#Vaka ve vefat artış miktarlarının hesaplanması 

vaka_artis = [0]

vefat_artis = [0]

iyilesen_artis = [0]



for i in range(len(data)-1):

    vaka_artis.append( data["Vaka_Sayisi"][i+1] - data["Vaka_Sayisi"][i] )

    vefat_artis.append( data["Vefat_Sayisi"][i+1] - data["Vefat_Sayisi"][i] )

    iyilesen_artis.append( data["Tedavi_Sayisi"][i+1] - data["Tedavi_Sayisi"][i] )



data["Vaka_Artış_Sayısı"] = vaka_artis

data["Vefat_Artış_Sayısı"] = vefat_artis

data["Tedavi_Artış_Sayısı"] = iyilesen_artis

data.info()
data.info()
#Son girilen Covid-19 verileri.

data.tail(1)
data
date_x     = data.Tarih

vaka_l     = data.Vaka_Sayisi

vefat_l    = data.Vefat_Sayisi

iyileşen_l = data.Tedavi_Sayisi

test_l     = data.Test_Sayisi



fgr = plt.figure(figsize=(20, 10), dpi=150, facecolor='w')

ax  = fgr.add_subplot(111)

ax.patch.set_facecolor('w')

ax.patch.set_alpha(1)



plt.plot(date_x,vaka_l,color='orange',linewidth = 2, alpha=1 ,label = "VAKA SAYISI");

plt.plot(date_x,vefat_l,color='red',linewidth = 2, alpha=1 ,label = "VEFAT SAYISI");

plt.plot(date_x,iyileşen_l,color='blue',linewidth = 2, alpha=1 ,label = "İYİLEŞEN SAYISI");

plt.plot(date_x,test_l,color='black',linewidth = 0.7, alpha=0.5 ,label = "GÜNLÜK YAPILAN TEST SAYISI");



plt.scatter(date_x,vaka_l,color='orange',linewidth = 0.5, alpha=1);

plt.scatter(date_x,vefat_l,color='red',linewidth = 0.5, alpha=1 );

plt.scatter(date_x,iyileşen_l,color='blue',linewidth = 0.5, alpha=1);

plt.scatter(date_x,test_l,color='gray',linewidth = 0.1, alpha=0.5);





plt.title('TÜRKİYE\'DEKİ GÜNCEL SON DURUM')

plt.xticks(rotation='vertical')

plt.xlabel('TARİH')

plt.ylabel('SAYI')

plt.legend(loc = 0)

plt.grid(color='black', linestyle="--", linewidth=0.5,alpha=0.5 ,dash_joinstyle = "bevel")

plt.show()
date_x     = data.Tarih

vaka_l     = data.Vaka_Artış_Sayısı

vefat_l    = data.Vefat_Artış_Sayısı

iyileşen_l = data.Tedavi_Artış_Sayısı

test_l     = data.Test_Sayisi



fig, ((ax1,ax4),(ax2,ax3)) = plt.subplots(2, 2, dpi=150, figsize=(20, 10),sharex='col')

fig.suptitle('GÜNLÜK ARTIŞ MİKTARLARI')



ax1.plot(date_x,vaka_l,'-o',color='orange',linewidth = 2, alpha=1)

ax2.plot(date_x,vefat_l,'-o',color='red',linewidth = 2, alpha=1)

ax3.plot(date_x,iyileşen_l,'-o',color='blue',linewidth = 2, alpha=1)

ax4.plot(date_x,test_l,'-o',color='blue',linewidth = 2, alpha=1)



ax1.set_xticklabels(date_x, rotation=90)

ax2.set_xticklabels(date_x, rotation=90)

ax3.set_xticklabels(date_x, rotation=90)

ax4.set_xticklabels(date_x, rotation=90)



ax1.grid(color='black', linestyle="--", linewidth=1,alpha=0.5 ,dash_joinstyle = "bevel")

ax2.grid(color='black', linestyle="--", linewidth=1,alpha=0.5 ,dash_joinstyle = "bevel")

ax3.grid(color='black', linestyle="--", linewidth=1,alpha=0.5 ,dash_joinstyle = "bevel")

ax4.grid(color='black', linestyle="--", linewidth=1,alpha=0.5 ,dash_joinstyle = "bevel")



ax1.set_title('VAKA SAYISI')

ax2.set_title('VEFAT SAYISI')

ax3.set_title('İYİLEŞEN SAYISI')

ax4.set_title('TEST SAYISI')

plt.show()