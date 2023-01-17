# Library'lerin yüklenmesi

import pandas as pd

import numpy as np 

import seaborn as sns

import matplotlib.pyplot as plt

import unidecode

import math

unidecode.unidecode('İ,Ö,Ü,Ş,Ç,Ğ,ı,ö,ü,ş,ç,ğ')

from scipy.stats.stats import pearsonr



# Veri dosyalarının yüklenmesi

df = pd.read_excel('../input/il-bazli-veri-220420/agg.xlsx')

df2 = pd.read_csv('../input/number-of-cases-in-the-city-covid19-turkey/number_of_cases_in_the_city.csv')



# Veri setlerinin kopyalanması

ilbazlimetrik = df

sehir_covid = df2



# Veri setlerine hızlı bir bakış

ilbazlimetrik.head()
# Veri setlerine hızlı bir bakış

sehir_covid.head()
# Veri setinin manipülasyonu

ilbazlimetrik = ilbazlimetrik[ilbazlimetrik['Sehir']!="Turkiye"]

sehir_covid = sehir_covid[['Province','Number of Case']]

ilbazlimetrik['Sehir']=ilbazlimetrik['Sehir'].apply(lambda x: unidecode.unidecode(x))



# Tabloların birleştirilmesi

agg = pd.merge(ilbazlimetrik,sehir_covid,'left',left_on=ilbazlimetrik['Sehir'],right_on=sehir_covid['Province'])

agg = agg.drop(['Province','Sehir'],axis=1)
# Birleştirilen tablolara göz atılması

agg.head()
# Yeni tablodaki hesaplamaların yapılması

agg['Olum_sayisi_binkisi']= agg['Olum_sayisi']/agg['Nufus']*1000

agg['Otomobil_sayisi_binkisi']=agg['Otomobil_sayisi']/agg['Nufus']*1000

agg['Hastane_binkisibasi']=agg['Toplam_hastane']/agg['Nufus']*1000

agg['Okuma_yazma_skor']=(agg['Okuma_yazma']-agg['Okuma_yazma'].min())/(agg['Okuma_yazma'].max()-agg['Okuma_yazma'].min())*100

agg['Vaka_binkisi'] = agg['Number of Case']/agg['Nufus']*1000

agg['Yatak_binkisibasi']=agg['Yatak_100binkisi']/100



# Nüfusa göre 750.000'den büyük illerin analize dahil edilmesi

agg2 = agg[agg['Nufus']>750000]

agg2 = agg2.drop(['Nufus','Okuma_yazma','Olum_sayisi','Otomobil_sayisi','Toplam_hanehalki','Toplam_hastane','Number of Case','Yatak_100binkisi'],axis=1)

agg2 = agg2.rename(columns={'key_0':'Sehir'})
agg2.head()
# Son tabloyla ilgili bir kaç grafik (il bazlı toplam 1000 kişi başı görülen vaka sayısı, nufus yogunluk)

plt.figure(figsize=(15,5))

plt.xticks(rotation=45)

sns.barplot(x='Sehir',y='Vaka_binkisi',data=agg2.sort_values('Vaka_binkisi',ascending=False),palette='RdBu')
corrMatrix = agg2.corr('pearson')

mask = np.triu(np.ones_like(corrMatrix, dtype=np.bool))

plt.figure(figsize=(15,10))

sns.heatmap(corrMatrix,annot=True, mask=mask, cmap='coolwarm',center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})