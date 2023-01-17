import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

import seaborn as sns

import sys

from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS

ogr = pd.read_csv('/kaggle/input/yok-20182019/Ogrenci_Sayilari_2018_2019_.csv')
ogr.rename(columns = {'Unnamed: 1':'universite_adi','Unnamed: 2':'tur','Unnamed: 3':'il',

'Unnamed: 4':'ogrenim_turu','Unnamed: 5':'onlisans_erkek','Unnamed: 6':'onlisans_kadin',

'Unnamed: 7':'onlisans_toplam','Unnamed: 8':'lisans_erkek','Unnamed: 9':'lisans_kadin',

'Unnamed: 10':'lisans_toplam','Unnamed: 11':'ylisans_kadin','Unnamed: 12':'ylisans_erkek',

'Unnamed: 13':'ylisans_toplam','Unnamed: 14':'doktora_erkek','Unnamed: 15':'doktora_kadin',

'Unnamed: 16':'doktora_toplam','Unnamed: 17':'genel_erkek','Unnamed: 18':'genel_kadin',

'Unnamed: 19':'genel_toplam',} , inplace = True)
df_ogr = pd.DataFrame(ogr)

df_ogr.head()
df_ogr.set_index("universite_adi" , inplace = True)
df_ogr.drop(['Unnamed: 0','Unnamed: 20'], axis=1 ,inplace = True)
df_ogr.drop(df_ogr.index[[0,1,2,26,27]] , inplace = True)
df_ogr.info()
df_ogr.describe()
df_ogr['il'].value_counts().head(10)

df_ogr['ogrenim_turu'].value_counts()
df_ogril = df_ogr['il'].head(160)

sns.countplot(x = df_ogril)

plt.title('İllere Göre Üniversite Sayıları')

plt.xticks(rotation = 90)

plt.show()
sns.countplot(x = df_ogr['ogrenim_turu'])

plt.title('Öğrenim Türüne Göre Üniversite Sayıları')

plt.xticks(rotation = 90)

plt.show()
sns.countplot(x = 'tur', hue='ogrenim_turu' ,data = df_ogr)

plt.title('Üniversite Türüne Göre Üniversite Sayıları')

plt.xticks(rotation = 90)

plt.show()
plt.figure(figsize = (10,10))

stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='black',

                          stopwords=stopwords,

                          max_words=1200,

                          max_font_size=120, 

                          random_state=42

                         ).generate(str(df_ogr['il']))

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()