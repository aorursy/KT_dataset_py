# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
netflix=pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')

netflix.shape
netflix.head(3)
netflix.info()
netflix.isnull().sum()
netflix['director'].fillna('unknown',inplace=True)

netflix['cast'].fillna('unknown',inplace=True)

netflix['country'].fillna('unknown',inplace=True)

netflix['rating'].fillna(netflix['rating'].mode()[0],inplace=True)

netflix.drop(['date_added'],axis=1,inplace=True)
netflix.isnull().sum().sum()
netflix['type'].value_counts()
import seaborn as sns

import matplotlib.pyplot as plt

sns.countplot(netflix['type'])
netflix['type'].value_counts().plot.pie(autopct='%1.1f%%',figsize=(10,5))
import re

def remove_text(text):

    text=re.sub("\D", "", text)

    return text

netflix['duration']=netflix['duration'].apply(lambda x:remove_text(x))
netflix['duration'].head()
movies=netflix[netflix['type']=='Movie']

tv_shows=netflix[netflix['type']=='TV Show']
netflix['release_year'].hist()
sns.kdeplot(movies['release_year'],color='g',shade=True,label='movies')

sns.kdeplot(tv_shows['release_year'],color='y',shade=True,label='TV Shows')
sns.kdeplot(movies['duration'],color='r',shade=True,label='movies')
sns.kdeplot(tv_shows['duration'],color='b',shade=True,label='TV Shows')
movies['rating'].value_counts()
movies['rating'].value_counts()[:5].plot(kind='bar')
tv_shows['rating'].value_counts()[:5].plot(kind='bar')
netflix['listed_in'].value_counts()[:10].plot(kind='barh')
movies['listed_in'].value_counts()[:10].plot(kind='barh',color='r')
tv_shows['listed_in'].value_counts()[:10].plot(kind='barh',color='pink')
movies['country'].value_counts()[:10].plot(kind='barh',color='green')
tv_shows['country'].value_counts()[:10].plot(kind='barh',color='brown')
movies['director'].value_counts()[1:11].plot(kind='bar')
tv_shows['director'].value_counts()[1:11].plot(kind='barh')
def get_director(director):

     return netflix.loc[netflix['director']==director,['title','release_year','listed_in']]

get_director('Steven Spielberg')    
def movies_shows(data,year):

    return data.loc[data['release_year']==year,['title']].head()

movies_shows(movies,2019)
movies_shows(tv_shows,2010)
from wordcloud import WordCloud

plt.subplots(figsize=(25,8))

wordcloud = WordCloud(

                          background_color='Black',

                          width=1920,

                          height=1080

                         ).generate(" ".join(movies['title']))

plt.imshow(wordcloud)

plt.axis('off')

#plt.savefig('cast.png')

plt.show()
plt.subplots(figsize=(25,8))

wordcloud = WordCloud(

                          background_color='Black',

                          width=1920,

                          height=1080

                         ).generate(" ".join(tv_shows['title']))

plt.imshow(wordcloud)

plt.axis('off')

#plt.savefig('cast.png')

plt.show()
plt.subplots(figsize=(25,8))

wordcloud = WordCloud(

                          background_color='Black',

                          width=1920,

                          height=1080

                         ).generate(" ".join(netflix['director']))

plt.imshow(wordcloud)

plt.axis('off')

#plt.savefig('cast.png')

plt.show()
plt.subplots(figsize=(25,8))

wordcloud = WordCloud(

                          background_color='Black',

                          width=1920,

                          height=1080

                         ).generate(" ".join(netflix['cast']))

plt.imshow(wordcloud)

plt.axis('off')

#plt.savefig('cast.png')

plt.show()