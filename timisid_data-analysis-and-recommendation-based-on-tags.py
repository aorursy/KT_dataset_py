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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
df1 = pd.read_csv("../input/ted-talks/ted_main.csv")

df1.head(1)
import datetime

df1['film_date'] = df1['film_date'].apply(lambda x: datetime.datetime.fromtimestamp( int(x)).strftime('%d-%m-%Y'))

df1['published_date'] = df1['published_date'].apply(lambda x: datetime.datetime.fromtimestamp( int(x)).strftime('%d-%m-%Y'))
df1.info()
df1.nunique()
df1.isnull().sum()
df1.dropna(subset=['speaker_occupation'],inplace=True)
df1.reset_index(inplace=True)
df1_new = df1[['comments', 'event', 'main_speaker','title', 'speaker_occupation', 'views', 'published_date']]
df1_new.head()
fig,ax=plt.subplots(figsize=(17,5))

a=sns.barplot(y=df1_new['speaker_occupation'].value_counts(ascending=False).head(15).index, 

              x=df1_new['speaker_occupation'].value_counts(ascending=False).head(15).values, ax=ax, palette='afmhot')

a.set(title="top 15 speaker's occupation")
fig,ax=plt.subplots(figsize=(17,6))

a=sns.barplot(y=df1_new['main_speaker'].value_counts(ascending=False).head(10).index, 

              x=df1_new['main_speaker'].value_counts(ascending=False).head(10).values, ax=ax, palette='afmhot')

a.set(title="top 10 speakers perform more than once")



df1_new[df1_new.main_speaker=='Hans Rosling'].sort_values(by='views')
fig,ax=plt.subplots(figsize=(17,5))

b=sns.barplot(y=df1_new['event'].value_counts(ascending=False).head(15).index, 

            x=df1_new['event'].value_counts(ascending=False).head(15).values, ax=ax, palette='afmhot')

b.set(title='top 15 most held yearly event')
df1_new['year']=df1_new['published_date'].apply(lambda x: x[-4:])

ig,ax=plt.subplots(figsize=(17,7))

sns.lineplot(x=df1_new['year'].value_counts().index,y=df1_new['year'].value_counts().values, marker='o')

sns.set_style('darkgrid')
fig,ax=plt.subplots(figsize=(17,7))

d=sns.barplot(x=df1_new[df1_new['views']>1000000].sort_values(by='views', ascending=False).head(10)['views'],

              y=df1_new[df1_new['views']>1000000].sort_values(by='views', ascending=False).head(10)['title'],palette='afmhot', ax=ax)

d.set(xlim=(15000000,50000000))

d.set(title='top 10 most watched based on title')
fig,ax=plt.subplots(figsize=(17,7))

e=sns.barplot(y=df1_new.sort_values(by='comments', ascending=False)['title'].head(10).values,

            x=df1_new.sort_values(by='comments', ascending=False)['comments'].head(10).values, palette='afmhot', ax=ax)

e.set(title='top 10 most commented based on title')

import re
def clean_text(x):

    letter_only=re.sub("[^a-zA-Z]", " ", x)

    return ' '.join(letter_only.split()).lower()
df1_new['tags']=df1['tags']

df1_new.tags=df1_new.tags.astype('str')
df1_new['tags']=df1_new['tags'].apply(clean_text)
df1_new.head(1)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

cv_tags=cv.fit_transform(df1_new['tags'])

df_genres=pd.DataFrame(cv_tags.todense(), columns=cv.get_feature_names(), index=df1_new['title'])
from sklearn.metrics.pairwise import cosine_similarity
cos_sim=cosine_similarity(cv_tags)
def get_recommendation_based_title(x):

    index_to_search = df1_new[df1_new['title']==x].index[0]

    series_similar=pd.Series(cos_sim[index_to_search])

    index_similar=series_similar.sort_values(ascending=False).head(10).index

    return df1_new.loc[index_similar]
get_recommendation_based_title('Do schools kill creativity?')
def get_recommendation_based_speakers(x):

    index_to_search = df1_new[df1_new['main_speaker']==x].index[0]

    series_similar=pd.Series(cos_sim[index_to_search])

    index_similar=series_similar.sort_values(ascending=False).head(10).index

    return df1_new.loc[index_similar]
get_recommendation_based_speakers('Hans Rosling')
def get_recommendation_based_speaker_occupation(x):

    index_to_search = df1_new[df1_new['speaker_occupation']==x].index[0]

    series_similar=pd.Series(cos_sim[index_to_search])

    index_similar=series_similar.sort_values(ascending=False).head(10).index

    return df1_new.loc[index_similar]
get_recommendation_based_speaker_occupation('Artist')