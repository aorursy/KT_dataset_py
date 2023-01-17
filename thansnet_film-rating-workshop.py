import pandas as pd
film_listesi=pd.read_csv('../input/imdbratings.data')
film_listesi.head()
#Tipini öğrenelim
type(film_listesi)
#Tipini öğrenelim
type(film_listesi.title)
film_listesi.head()
#liste hakkında bilgi alalım
film_listesi.info()
#Sıralayalım
film_listesi.title.sort_values(ascending=False)
#Bağıntısına bakalım
film_listesi.corr()
film_listesi.title.sort_values(ascending=False)
#title a göre sıralama işlemi
film_listesi.sort_values('title')
#süreye göre sıralama işlemi

#süre ve türe göre sıralayalım
film_listesi.sort_values(['duration','genre'])
film_listesi.sort_values(['star_rating','genre','duration'])
#aynı datayı yeniden kullanalım fakat başka bir değişkene atayalım
film_listesi2=pd.read_csv('../input/imdbratings.data')
#Film süresine göre uzunluk kategorileyelim
film_listesi2_kategori=[]
for i in film_listesi2.duration:
    if i <=80 :
        film_listesi2_kategori.append('Çok Kısa')
    elif i > 80 and i<=120:
        film_listesi2_kategori.append('Normal')
    else:
        film_listesi2_kategori.append('Çok Uzun')
        

film_listesi2['Uzunluk_Kategori']=film_listesi2_kategori
film_listesi2.head(100)
#Sıralayalım ascending=False
film_listesi2.sort_values(['Uzunluk_Kategori','duration'])
#sadece kategorisi normal olanları listeleyelim
film_listesi2[film_listesi2.Uzunluk_Kategori=='Normal']
film_listesi2[film_listesi2.genre=='Action']
film_listesi2[(film_listesi2.Uzunluk_Kategori=='Normal') & (film_listesi2.genre=='Action')]
film_listesi2[film_listesi2.genre=='Action'][film_listesi2.Uzunluk_Kategori=='Normal']
#rating değeri 9.0 dan büyük olanlar
film_listesi2[(film_listesi2.genre=='Adventure')|(film_listesi2.star_rating>=9.0)]
film_listesi2.groupby('genre').star_rating.mean().sort_values(ascending=False)
%matplotlib inline

film_listesi2.groupby('genre').star_rating.mean().sort_values(ascending=False).plot(kind='line')
film_listesi2.groupby('genre').star_rating.mean().sort_values(ascending=False).plot(kind='line')
film_listesi2.groupby('genre').star_rating.mean().sort_values(ascending=False).plot(kind='bar')
film_listesi2.groupby('genre').star_rating.agg(['mean','median','min','max','count']).sort_values('mean',ascending=False)
film_listesi2.info()
film_listesi2[film_listesi2.actors_list.str.contains('John Travolta')]
film_listesi2.actors_list.str.replace('[','').str.replace(']','')
film_listesi2.actors_list=film_listesi2.actors_list.str.replace("u'",'').str.replace('[','').str.replace(']','')

actor_series=film_listesi2.actors_list.to_frame()
film_listesi2.head()
film_listesi2.actors_list=film_listesi2.actors_list.str.replace("'",'')
film_listesi2.head()

import os
os.mkdir('../output')
film_listesi2.to_excel('../output/filmler.xlsx')
pd.read_excel('../output/filmler.xlsx')
