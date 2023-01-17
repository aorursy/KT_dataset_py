# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import random

import matplotlib.pyplot as plt # data visualization library

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS #used to generate world cloud

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data= pd.read_csv('../input/movies.csv')

data.shape

#Satır-sütun sayısı
data.head
#Head() fonksiyonu varsayılan olarak veri setinin ilk 5 satırını görüntülüyor.

data.head() 
#Tail fonksiyonu veri setinin son 5 satırını görüntülüyor.

data.tail() 
#Veri hakkında genel bilgi verir.

data.info() 
#movie.csv

data.shape
#info() metodu ile eksik veriye sahip sütunlar hakkında bilgiyi bu şekilde edindik. 

#Daha anlaşılır bir şekilde eksik veriler hakkında bilgi edinmek için 

data.isnull().sum().sort_values(ascending=False)
#ratings.csv

ratings_data.shape
ratings_data.isnull().any()
tags_data=pd.read_csv('../input/tags.csv',sep=',')

tags_data.shape
tags_data.isnull().any()
#Satır veya sütunda null veri varsa bunu silmek için dropna() fonksiyonu kullanılır.

#Ya da eksik verileri doldurmak için fillna() fonksiyonu kullanılabilir.

tags_data=tags_data.dropna()
#Null verileri sildikten sonra tekrar kontrol ediyoruz.Tags.csv dosyasında boş dosya görünmemekte

tags_data.isnull().any()
#Veri setinde kaç farklı etiket(tag) var?

unique_tags=tags_data['tag'].unique().tolist()

len(unique_tags)
#Sütunların veri tipleri hakkında bilgi almak için (64-bit integer, float, object vs)

data.dtypes  
#Veri içerisinde kaç farklı movieId var?

movies = data['movieId'].unique().tolist()

len(movies)
#Rating.csv'yi okumak için

ratings_data=pd.read_csv('../input/ratings.csv',sep=',')

ratings_data.shape

#Satır x Sütun sayısı için yine data.shape kullanılıyor.
#describe() metodu sayısal verilere sahip olan sütunların max, min , std…gibi istatiksel değerlerini döndürür. 

#Rating sütununun istatiksel özetini görebilmek için

ratings_data.describe()
#Filme verilen en yüksek puan

ratings_data['rating'].max()
#Filme verilen en düşük puan

ratings_data['rating'].min()
#Veri setinde filtreleme işlemi

#Drama filmlerinin listesini almak için filtreleme

drama_movies=data['genres'].str.contains('Drama')

data[drama_movies].head()
#Drama filmlerinin sayısını listelemek için

drama_movies.shape
#İstenilen tag e göre filtreleme yapmak için

tag_search = tags_data['tag'].str.contains('dark')

tags_data[tag_search].head()
#Elimizde birden fazla DataFrame, Series olduğunda bunlar birleştirilmek istenebilir. Bu durumda "merge" veya "concat" kullanılmaktadır.

movie_data_ratings_data=data.merge(ratings_data,on = 'movieId',how = 'inner')

movie_data_ratings_data.head(3)
#Yüksek puanlı filmleri görüntülemek için

#Puanı 4'den yüksek olan ilk 10 film görüntülendi.

high_rated= movie_data_ratings_data['rating']>4.0

movie_data_ratings_data[high_rated].head(10)
#Düşük puanlı filmleri görüntülemek için

#Puanı 4'den küçük olan ilk 5 film görüntülendi.

low_rated = movie_data_ratings_data['rating']<4.0

movie_data_ratings_data[low_rated].head()
#Benzersiz film türü sayısı görüntülendi.

unique_genre=data['genres'].unique().tolist()

len(unique_genre)
#Sıralama işlemi için "sort" kullanılmaktadır.

#En çok oy alan 25 film görüntülendi.

most_rated = movie_data_ratings_data.groupby('title').size().sort_values(ascending=False)[:25]

most_rated.head(25)
#movies.csv den yalnızca başlık ve tür sütunları listelendi. 

data[['title','genres']].head()
#Her türün görünme sayısını hesaplayan bir fonksiyon yazıldı.

def count_word(df, ref_col, liste):

    keyword_count = dict()

    for s in liste: keyword_count[s] = 0

    for liste_keywords in df[ref_col].str.split('|'):

        if type(liste_keywords) == float and pd.isnull(liste_keywords): continue

        for s in liste_keywords: 

            if pd.notnull(s): keyword_count[s] += 1

    # convert the dictionary in a list to sort the keywords  by frequency

    keyword_occurences = []

    for k,v in keyword_count.items():

        keyword_occurences.append([k,v])

    keyword_occurences.sort(key = lambda x:x[1], reverse = True)

    return keyword_occurences, keyword_count
#Türlerin sayımı yapıldı.

genre_labels = set()

for s in data['genres'].str.split('|').values:

    genre_labels = genre_labels.union(set(s))
#Her bir türün kaç defa görüntülendiği count_word() fonksiyonuyla sayıldı.

keyword_occurences, dum = count_word(data, 'genres', genre_labels)

keyword_occurences
#Kelimelerin rengini kontrol eden fonksiyon yazıldı.

def random_color_func(word=None, font_size=None, position=None,

                      orientation=None, font_path=None, random_state=None):

    h = int(360.0 * tone / 255.0)

    s = int(100.0 * 255.0 / 255.0)

    l = int(100.0 * float(random_state.randint(70, 120)) / 255.0)

    return "hsl({}, {}%, {}%)".format(h, s, l)

words = dict()

trunc_occurences = keyword_occurences[0:50]

for s in trunc_occurences:

    words[s[0]] = s[1]

tone = 100 # define the color of the words

f, ax = plt.subplots(figsize=(14, 6))

wordcloud = WordCloud(width=550,height=300, background_color='purple', 

                      max_words=1628,relative_scaling=0.7,

                      color_func = random_color_func,

                      normalize_plurals=False)



wordcloud.generate_from_frequencies(words)

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()

##Sonuç, wordcloud olarak gösterilmektedir.



#Aynı sonuç Sütun Grafik şeklinde gösterildi.

fig = plt.figure(1, figsize=(18,13))

ax2 = fig.add_subplot(2,1,2)

y_axis = [i[1] for i in trunc_occurences]

x_axis = [k for k,i in enumerate(trunc_occurences)]

x_label = [i[0] for i in trunc_occurences]

plt.xticks(rotation=85, fontsize = 15)

plt.yticks(fontsize = 15)

plt.xticks(x_axis, x_label)

plt.ylabel("No. of occurences", fontsize = 24, labelpad = 0)

ax2.bar(x_axis,y_axis,color=['purple', 'red', 'green', 'blue', 'cyan'],

        edgecolor=["blue","green","cyan","black","red"])

plt.title("Popularity of Genres",bbox={'facecolor':'k', 'pad':5},color='w',fontsize = 30)

plt.show()