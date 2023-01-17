# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/youtube-new/USvideos.csv") #Dataset'i okuma
data.head() # İlk 5 data
#Birkaç gereksiz şeyi silme

data.drop(["video_id"],axis = 1,inplace=True)

data.drop(["trending_date"],inplace=True,axis = 1)

data.drop(["publish_time"],inplace=True,axis = 1)

data.drop(["thumbnail_link"],inplace=True,axis = 1)

data.drop(["comments_disabled"],inplace=True,axis = 1)

data.drop(["ratings_disabled"],inplace=True,axis = 1)

data.drop(["video_error_or_removed"],inplace=True,axis = 1)
data
len(data.index) #index sayısı
len(data.columns) # column sayısı
data["likes"].mean() # Beğeni ortalaması
data["dislikes"].mean() # Dislike ortalaması
#en yüksek views alan videonun title' ı 

data["title"][(data["views"]) == (data["views"].max())].iloc[0]
#en yüksek views alan videonun görüntüleme sayısı

data["views"].max()
#en düşük views alan videonun title' ı 

data["title"][(data["views"]) == (data["views"].min())].iloc[0]
#en yüksek views alan videonun görüntüleme sayısı

data["views"].min()
#category_id'ye göre comment_count ortalaması

data.groupby("category_id").mean()[["comment_count"]]
#Kategorilerde kaç adet Video var

data["category_id"].value_counts()
#Her videonun başlık uzunluğunu (title_lenght) içeren sutün ekleme

data["title_lenght"]=data["title"].apply(len)
data
# Her bir videonun etiket sayısını(label_number) içeren sutün ekleme

# Her tag | ile ayrılıyor

def NumberTag(tag):

    tagList = tag.split("|")

    return len(tagList)

data["label_number"] = data["tags"].apply(NumberTag)
data
#iteritems(): Her bir index'i ve o index'e karşılık gelen değeri alarak bir tane tuple'a(demete) yerleştirir.

def likesRates(likes,dislikes):

    likesList = list() #like'larin listesi

    for key,value in list(likes.iteritems()):

        likesList.append(value)

    dislikesList = list() # dislike'ların listesi

    for key,value in list(dislikes.iteritems()):

        dislikesList.append(value)

    likedislike = list(zip(likesList,dislikesList)) 

    rates = list()

    for like,dislike in likedislike:

        if(like + dislike) == 0: #ZeroDivisionError hatası almamak için

            rates.append(0)

        else: 

            rates.append(like / (like + dislike))

   

    return rates

data["likes_dislikes"] =  likesRates(data["likes"],data["dislikes"])
data.sort_values(["likes_dislikes"],ascending=False,inplace = True)

data