import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline

import warnings

warnings.filterwarnings('ignore') 

sns.set()



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/birinci_dunya_savasi_kastamonu_sehit_bilgileri.csv")
data.columns
data.head()
data.info()
# 1.Dünya Savaşı Kastamonu Toplam Şehit Sayısı # 1704

print(data.S_NO.value_counts())
data["OLUM_TARIHI"] = data['OLUM_TARIHI_GUN'].map(str) + '-' + data['OLUM_TARIHI_AY'].map(str) + '-' + data['OLUM_TARIHI_YIL'].map(str)

data["BIRLIK_NO"] = data['BIRLIK_NO_1'].map(str) + '-' + data['BIRLIK_NO_2'].map(str) + '-' + data['BIRLIK_NO_3'].map(str) + '-' + data['BIRLIK_NO_4'].map(str) + '-' + data['BIRLIK_NO_5'].map(str) + '-' + data['BIRLIK_NO_6'].map(str) + '-' + data['BIRLIK_NO_7'].map(str)

columns = ["OLUM_TARIHI_GUN", "OLUM_TARIHI_AY", "OLUM_TARIHI_YIL", "BIRLIK_NO_1", "BIRLIK_NO_2", "BIRLIK_NO_3", "BIRLIK_NO_4", "BIRLIK_NO_5", "BIRLIK_NO_6", "BIRLIK_NO_7"]

data = data.drop(columns, axis=1).reset_index(drop = True)
arslanbey_sehit_liste = data[data.KOY == "ARSLANBEY"]

arslanbey_sehit_liste.head()
data.ILCE.unique()
data.ILCE.fillna("DIGER",inplace = True)

data.ILCE.replace([' '],"DIGER",inplace = True)
ilce_list = data.ILCE.unique()
result = []

for i in ilce_list:

    x = data[data['ILCE']==i]

    count = len(x)

    result.append(count)

new_data = pd.DataFrame({'ilce_list': ilce_list,'count':result})

new_index = (new_data['count'].sort_values(ascending=False)).index.values

sorted_data = new_data.reindex(new_index)
plt.figure(figsize=(12,9))

sns.barplot(x=sorted_data['ilce_list'], y=sorted_data['count'])

plt.xticks(rotation= 45)

plt.xlabel('İlçeler')

plt.ylabel('Şehit Sayısı')

plt.title('İlçe Bazlı Şehit Sayısı')
data.columns
data.RUTBE.fillna("DIGER",inplace = True)

data.RUTBE.replace(['Yd.Sb.'],"YD.SB.",inplace = True)
rutbe_list = list(data.RUTBE.unique())

rutbe_list
rutbe_data = data.RUTBE.value_counts()

plt.figure(figsize=(25,8))

sns.barplot(x=rutbe_data.index,y=rutbe_data.values)

plt.ylabel('Şehit Sayısı')

plt.xlabel('Rütbeler')

plt.title('Rütbe Bazlı Şehit Sayısı',color = 'blue',fontsize=15)