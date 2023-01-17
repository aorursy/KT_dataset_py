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
data = pd.read_csv("../input/youtube-new/USvideos.csv")

data.info()
# Info about data
data.corr()
#comment_count and views 
#views and likes
#comment_count and dislikes 
#Corr map
Data_Corr_Map = plt.subplots(figsize=(15,15))
sns.heatmap(data.corr(), annot = True, linewidths = 5, fmt= '.2f')
plt.show()
# Heat map for finding correlations 
# If corr. close to 1 it is right proportion
# If corr. close to -1 it is inverse proportion
# If corr. close to 0 there are no proportions

data.columns
# Features of the data


data.head(10)
# First 10 videos features
# Line Plot
data.views.plot(kind = 'line', color = 'red', label = 'Views', linewidth = 1, alpha = 0.3,
               grid = True, linestyle = ':')
data.likes.plot(kind = 'line', color = 'g', label = 'likes', linewidth = 1, alpha = 0.3,
                 linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()



# Scatter Plot
plt.scatter(data.likes, data.dislikes, alpha = 0.1, color = "red")
plt.xlabel('Likes')
plt.ylabel('Dislikes')
plt.title('Likes & Dislikes Plot')
plt.show()
# Histogram
data.likes.plot(kind = 'hist', bins =100, figsize = (10,10))
plt.show()
# Filtering
x = data['likes'] > 1000000
data[x]
data[(data['likes'] > 1000000) & (data['views'] > 30000000)]
# While and For Loops

# While
i = 2
while i < 20:
    print('i : ', i)
    i = i**2
    print('i is bigger than 20')
    
# For
liste = ["ahmet","mehmet","ali","ayse","veli"]
for each in liste:
    if(len(each) >= 4):
        print(each)
 
dictionary1 = {'FB':'Fenerbahce','GS':'Galatasaray','BJK':'Besiktas'}
for key,value in dictionary1.items():
    print(key," : ",value)
print('')

# Functions:

# User Defined Func:


def vucut_index(kilo,boy):
    """ input = boy and kilo
        output = body index """
    output = kilo / boy**2
    print("Vucud Kitle Indeksiniz :", output)

print(vucut_index(20,2))

from math import sqrt

def hipotenus_hesaplama(a,b):
    c = sqrt((a**2 + b**2))
    print("hipotenus", c)

print(hipotenus_hesaplama(5,12))    
    
# Nested Func:
def saglik_hesaplama(k,b,y,s):
    
    def vucut_index(k,b):
        """ input = boy and kilo
        output = body indeks """
    
        vucut_index = k / b**2
    
        return vucut_index
    
    def yas_spor(y,s):
        """ input = yas and spor per day (hour)
        output = healt situation """
        
        if(y<15 and s>1):
             c = 1
        elif( 15 <y< 35 and s>=2):
             c = 1
        elif(35<y<65 and 3>s>1):
             c = 1
        elif(y>65 and s == 1):
             c = 1
        else:
             c = 0
        return c
    
    
        
    if(18,5< vucut_index(k,b) <29,9 & yas_spor(y,s) == 1):
        
        sonuc = "saglikli"
    else:
        sonuc = "sagliksiz"
        

    print("Saglik Durumunuz :", sonuc)
     
print(saglik_hesaplama(80,180,17,2))
            
                 
    
    
    
# Default and Flexible Func:

def yas_hesaplama(dt, yil=2020):
    yas = yil-dt
    return yas

print(yas_hesaplama(2003,2020))

print(yas_hesaplama(2003,2021))
    
def hafiza_testi(*args):
    ''' Input = Bildiginiz Asal Sayilar
    Output = Matematik Yeterliligi '''

    if len(args)>50: 
        print("Matematiginiz Cok iyi")
    elif len(args)>= 25:
        print("Matematiginiz iyi")
    else:
        print("Matematiginiz yetersiz")
        

        
        
# Lambda Func:
ucgen_alan = lambda a,h: a*h/2

print(ucgen_alan(5,6))
# Anonymous Func:
list3 = [5,13,42,54,90]
y = map(lambda x:x%5, list3)
print(list(y))

#Iterators

city = "Berlin"
iterator1 = iter(city)
print(next(iterator1))
print(*iterator1)

# List Comprehension

liste1 = [50,43,23,35,23,65,72,98,104,3443]
liste2 = [i/5 if i%5 == 0 else i**2/10 if len(str(i))>=3 else "bu sayi 2k" if i%2 == 0 else i for i in liste1]
print(liste2)
data.info() 
data.head()
# List Comprehension w/Data
ortalama = sum(data.views)/len(data.views)
print(ortalama)
data["Mean_Lvl"] = ["High" if i > ortalama else "low" for i in data.views]
data.loc[:5,["Mean_Lvl","views"]]
# Exploratory Data Analysis
print(data.channel_title.value_counts(dropna =False))
# Hangi kanalin kac videosu oldugunu gosterir.

# Visual Exploratory

data.boxplot(column='dislikes', by = 'comments_disabled')
plt.show()
# Yorumlar acik ve kapaliykenki dislike oranlari (mediana gore)
# Gorunuse gore yorumlari kapali videolar like dislikei da kapali videolar
# Melting

data_melting = data.head(10)  
melted_data = pd.melt(frame=data_melting, id_vars = 'title', value_vars= ['likes','dislikes'])
melted_data
# id_vars == degismemesini istedigin veri
# value_vars == degismesini istedigin veriler

# Pivoting 
# Melted veriyi eski haline getirir

melted_data.pivot(index='title', columns = 'variable',values= 'value')
# Concatenating Data

data1 = data.likes.head()
data2 = data.dislikes.head()
concated_data = pd.concat([data1,data2],axis =1)
concated_data
# Changing Data Types

data.dtypes
data.category_id = data.category_id.astype('float')
data.dtypes

# Assert
# Assert dogru mu degil mi diye kontrol amaciyla kullanilir
assert 25/5==5
# Dogruysa error vermez yanlis ise error verir
# Building Data Frames

Name = ["Rekkles","BrokenBlade","Faker"]
Team = ["Fnatic","TSM","T1"]
Role = ["Bottom","Top","Mid"]
Tear = ["High","Mid","Legendary"] 
list_label = ["Name","Team","Role","Tear"]
list_col = [Name,Team,Role,Tear]
zipped = list(zip(list_label,list_col))
data_dic = dict(zipped)
df = pd.DataFrame(data_dic)
df


# Adding new column
df["Best Champs"] = ["Tristana","Irelia","Ryze"]
df
# Yeni bir feature olusturmak tek degerle tum datalara

df["Gender"] = "Male"
df
data.head()
data.head()
data.info()
# Hangi kanalin kac videosu var
data.channel_title.value_counts()
# Amac = Videolarinin izlenmesine gore top 15 kanal

ortalama_list = []
deneme_data = data
list_kanallar = list(deneme_data.channel_title.unique()) 

for i in list_kanallar:
    datax = data[data['channel_title'] == i] 
    print(datax.head())
    ortalama = round(sum(datax.views)/len(datax))
    ortalama_list.append(ortalama)
    
dfx = pd.DataFrame({"kanal adi: ": list_kanallar, "Ortalama Izlenme ": ortalama_list }).sort_values(by=["Ortalama Izlenme "],ascending=False)
dfx = dfx.head(15)
   
plt.figure()
sns.barplot(x= dfx["kanal adi: "], y= dfx["Ortalama Izlenme "], palette = sns.cubehelix_palette(15))
plt.xticks(rotation= 90) # X ekseninde 90 derecelik aciyla statesleri yazd
plt.xlabel('Kanallar')
plt.ylabel('Ortalama')
plt.title('Video Ortalamasina Gore Top 15 Kanal') 
plt.show()
# Hangi kanalin max likei var
like_list = []
list_kanallar = list(deneme_data.channel_title.unique())
deneme_data = data

for i in list_kanallar:
    data_like = data[data['channel_title'] == i] 
    likes = sum(data_like.likes)
    #print(data_like.head())
    like_list.append(likes)
    
df_like = pd.DataFrame({"Kanallar": list_kanallar,"Like Sayisi": like_list}).sort_values(by=["Like Sayisi"],ascending=False)    
df_like = df_like.head(15)

plt.figure()
sns.barplot(x= df_like["Kanallar"], y= df_like["Like Sayisi"], palette = sns.cubehelix_palette(15), )
plt.xticks(rotation= 90) # X ekseninde 45 derecelik aciyla statesleri yazd
plt.xlabel('Kanallar')
plt.ylabel('Toplam Begenme Sayisi')
plt.title('Begenme Sayilarina Gore Top 15 Kanal') 
plt.show()



data.head()
max(data.views)
data2
dislikes
# Max Izlenen 15 videonun like disslike ve yorum sayisi 

sorted_data = data.views.sort_values(ascending=False)
data1 = sorted_data.head(15)
new_index = data1.index.values
data2 = data.reindex(new_index)
video_list = list(data2.trending_date)
likes = []
dislikes = []
comments = []

for i in video_list:
    x = data2[data2.trending_date == i]
    likes.append(x.likes)
    dislikes.append(x.dislikes)
    comments.append(x.comment_count)
    
f,ax = plt.subplots(figsize = (9,15))
sns.barplot(x=likes,y=video_list,color='blue',alpha= 0.5, label='Likes')
sns.barplot(x=dislikes,y=video_list,color='red',alpha= 0.5, label='Dislikes')
sns.barplot(x=comments,y=video_list,color='yellow',alpha= 0.5, label='Comments')

    
    







