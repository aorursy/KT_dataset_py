# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
anime = pd.read_csv('../input/anime.csv')

rating = pd.read_csv('../input/rating.csv')
anime.head()
rating.head()
user_id_list=[]

user_id_list=rating['user_id'].values

user_id_list=list(dict.fromkeys(user_id_list))

user_id_list[0:15]
anime_id_list=[]

anime_id_list=anime['anime_id'].values

anime_id_list=list(dict.fromkeys(anime_id_list))

anime_id_list.sort()

anime_id_list[0:10]
len(anime)
list_genre=[]

list_genre= anime['genre'].values

list_genre[0:10]
corpus=[]

for i in range(0,12294):

    first=list_genre[i]

    first = first.lower()

    first = first.replace(' ','')

    first = first.split(',')

    corpus.append(first)

print(corpus)
len(corpus)
unique_corpus = []

for x in corpus:

    if x not in unique_corpus:

        unique_corpus.append(x)
len(unique_corpus)
print(unique_corpus)