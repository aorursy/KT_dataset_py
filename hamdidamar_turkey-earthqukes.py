

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

import seaborn as sns 
deprem = pd.read_csv("/kaggle/input/turkey-earthquakes-2011-2019/Son-Depremler.csv")
deprem.head(10)
deprem.rename(columns = {'Region-Name':'Bölge','Date-Time':'Zaman','Mag':'Büyüklük','Type':'Türü','Latitude':'Enlem','Longitude':'Boylam','Depth':'Derinlik'} , inplace = True)

deprem.head()
deprem.info()
chosen_columns_deprem = ['Zaman','Büyüklük','Türü','Enlem','Boylam','Derinlik','Bölge']

df_deprem = pd.DataFrame(deprem,columns = chosen_columns_deprem)

df_deprem.head()
df_deprem['Zaman'].value_counts().head(10)
df_deprem['Derinlik'].value_counts().head(10)
df_deprem['Türü'].value_counts().head(10)
df_deprem['Enlem'].value_counts().head(10)
df_deprem['Boylam'].value_counts().head(10)
sns.countplot(deprem['Derinlik'].head(10))

plt.xticks(Rotation = 90)

plt.title('Derinliğe Göre Deprem Sayıları')

plt.show()
sns.countplot(deprem['Boylam'].head(10))

plt.xticks(Rotation = 90)

plt.title('Boylama Göre Deprem Sayıları')

plt.show()
sns.countplot(deprem['Enlem'].head(10))

plt.xticks(Rotation = 90)

plt.title('Enleme Göre Deprem Sayıları')

plt.show()
deprem.hist(color ='Yellow')
deprem.plot(subplots = True,color = 'Red')

plt.show()
f_buyukluk = (df_deprem.Büyüklük >= 3) & ((df_deprem.Bölge == "Turkey") 

| (df_deprem.Bölge == "Greece")| (df_deprem.Bölge == "Taiwan")| (df_deprem.Bölge == "Romania")

| (df_deprem.Bölge == "Myanmar")| (df_deprem.Bölge == "Hindu Kush Region, Afghanistan") )

data =df_deprem[f_buyukluk]

sns.countplot(x = data["Bölge"],data = data)

plt.xticks(Rotation = 90)

plt.title('Ülkelere Göre 3 ve Üzeri Büyüklüğündeki Deprem Sayıları')

plt.show()
f_buyukluk = (df_deprem.Büyüklük >= 3) & ((df_deprem.Bölge == "Cardak-Denizli") 

| (df_deprem.Bölge == "Cankaya-Ankara") | (df_deprem.Bölge == "Akhisar-Manisa")

| (df_deprem.Bölge == "Hisarcik-Kutahya")| (df_deprem.Bölge == "Marmaris-Mugla")

| (df_deprem.Bölge == "Simav-Kutahya") | (df_deprem.Bölge == "Dazkiri-Afyonkarahisar") 

| (df_deprem.Bölge == "Basmakci-Afyonkarahisar") | (df_deprem.Bölge == "Bozkurt-Denizli") 

| (df_deprem.Bölge == "Yesilova-Burdur") | (df_deprem.Bölge == "Soma-Manisa"))

data =df_deprem[f_buyukluk]

sns.countplot(x = data["Bölge"],data = data)

plt.xticks(Rotation = 90)

plt.title('İl ve İlçelere Göre 3 ve Üzeri Büyüklüğündeki Deprem Sayıları')

plt.show()
f_buyukluk = (df_deprem.Büyüklük >= 3) & ((df_deprem.Bölge == "Aegean Sea") 

| (df_deprem.Bölge == "Antalya Korfezi-AKDENIZ") | (df_deprem.Bölge == "Marmara Denizi (Bati)")

| (df_deprem.Bölge == "Off Coast of Oregon")| (df_deprem.Bölge == "Marmara Denizi (Orta)")

| (df_deprem.Bölge == "Marmara Denizi (Dogu)") | (df_deprem.Bölge == "Near Coast of Nicaragua") 

| (df_deprem.Bölge == "Near Coast of Ecuador") | (df_deprem.Bölge == "Near East Coast of Honshu, Japan") 

| (df_deprem.Bölge == "Santa Cruz Islands") | (df_deprem.Bölge == "Mariana Islands") 

| (df_deprem.Bölge == "Ryukyu Islands, Japan") 

| (df_deprem.Bölge == "Vanuatu Islands Region") | (df_deprem.Bölge == "Revilla Gigedo Islands Region") 

| (df_deprem.Bölge == "Crete, Greece"))

data =df_deprem[f_buyukluk]

sns.countplot(x = data["Bölge"],data = data)

plt.xticks(Rotation = 90)

plt.title('Ada ,Kıyı ve Bölgelere Göre 3 ve Üzeri Büyüklüğündeki Deprem Sayıları')

plt.show()