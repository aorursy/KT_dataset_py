# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
%matplotlib inline
# Any results you write to the current directory are saved as output.
con = sqlite3.connect('../input/database.sqlite')

artists = pd.read_sql('SELECT * FROM artists', con)
content = pd.read_sql('SELECT * FROM content', con)
genres = pd.read_sql('SELECT * FROM genres', con)
labels = pd.read_sql('SELECT * FROM labels', con)
years = pd.read_sql('SELECT * FROM years', con)
reviews = pd.read_sql('SELECT * FROM reviews', con)

con.close()
df1 = pd.merge(genres,reviews,on='reviewid')
df2 = pd.merge(labels,df1,on='reviewid')
df3 = pd.merge(content,df2,on='reviewid')
df_final = pd.merge(years,df3,on='reviewid')
df_final.head()
df_final.drop(['url','author_type','pub_date','pub_weekday','pub_day','pub_month'],axis=1,inplace=True)
df_final.dropna(inplace=True)
df_final['year'] = df_final['year'].astype(int)
df_final['pub_year'] = df_final['pub_year'].astype(int)
df_final.head()
df_final[df_final.duplicated('reviewid', keep=False)].sort_values('reviewid').head(5)
df_final.drop_duplicates('reviewid',inplace=True)
df_final.shape
df_final['text_len'] = df_final['content'].apply(lambda x: len(x))
df_final['text_words'] = df_final['content'].apply(lambda x: len(x.split()))
df_final['uniq_words'] = df_final['content'].apply(lambda x: len(set(x.split())))
df_final['lex_den'] = 100*df_final['uniq_words']/df_final['text_words']
df_final.head()
df_final[['text_len','text_words','uniq_words','lex_den']].hist(sharey=True, layout=(2, 2), figsize=(14, 12), color='b', alpha=.75, grid=False,bins=50);
df_final.author.value_counts()[:10].plot(kind='barh', figsize=(8,6));
plt.xlabel('Number of reviews',fontsize=16)
plt.ylabel('Author',fontsize=16)
plt.title('Top 10 Pitchfork writers',fontsize=16);
df_final[df_final.genre=='rock'].author.value_counts()[:10].plot(kind='barh', figsize=(8,6));
plt.xlabel('Number of reviews',fontsize=16)
plt.ylabel('Author',fontsize=16)
plt.title('Top 10 Pitchfork Rock writers',fontsize=16);
df_final['author'].value_counts()[:10].sum()/df_final['author'].value_counts().sum()
df_final[df_final['genre']=='rock']['author'].value_counts()[:10].sum()/df_final['author'].value_counts().sum()
df_final.groupby(['author','genre'])['text_words'].mean().sort_values(ascending=False)[:10].plot(kind='barh',figsize=(8,6));
plt.xlabel('Mean review length (longest)', fontsize=16)
plt.ylabel('Author, Genre', fontsize=16);
df_final[df_final['text_len']>100].groupby(['author','genre'])['text_words'].mean().sort_values(ascending=False)[-10:].plot(kind='barh',figsize=(8,6));
plt.xlabel('Mean review length (shortest)', fontsize=16)
plt.ylabel('Author, Genre', fontsize=16);
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8,6))
df_final.groupby(['author','genre'])['lex_den'].mean().sort_values(ascending=False)[:10].plot(kind='barh',figsize=(10,8),ax=ax1);
df_final[df_final['text_len']>100].groupby(['author','genre'])['lex_den'].mean().sort_values(ascending=False)[-10:].plot(kind='barh',figsize=(10,8),ax=ax2);
ax1.set_xlabel('Highest Mean Lexical Density');
ax2.set_xlabel('Lowest Mean Lexical Density');
ax1.set_ylabel('')
ax2.set_ylabel('');
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8,6))
df_final.groupby(['genre'])['text_words'].mean().plot(kind='barh',figsize=(10,8),ax=ax1);
df_final.groupby(['genre'])['lex_den'].mean().plot(kind='barh',figsize=(10,8),ax=ax2);
ax1.set_xlabel('Total words');
ax2.set_xlabel('Lexical Density');
ax1.set_ylabel('')
ax2.set_ylabel('');
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8,6))
df_final.groupby(['pub_year'])['text_words'].mean()[::-1].plot(kind='barh',figsize=(14,12),ax=ax1);
df_final.groupby(['pub_year'])['lex_den'].mean()[::-1].plot(kind='barh',figsize=(14,12),ax=ax2);
ax1.set_xlabel('Total words');
ax2.set_xlabel('Lexical Density');
ax1.set_ylabel('')
ax2.set_ylabel('');
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8,6))
df_final.groupby(['year'])['text_words'].mean().plot(figsize=(10,8),ax=ax1,color='b',alpha=.75);
df_final.groupby(['year'])['lex_den'].mean().plot(figsize=(10,8),ax=ax2,color='b',alpha=.75);
ax1.set_ylabel('Total words');
ax2.set_ylabel('Lexical Density');
ax1.set_xlabel('')
ax2.set_xlabel('Year of release');
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8,6))
df_final[(df_final.year>1980) & (df_final.year<1990)].groupby(['year'])['text_words'].mean().plot(figsize=(10,8),ax=ax1,color='b',alpha=.75);
df_final[(df_final.year>1980) & (df_final.year<1990)].groupby(['year'])['lex_den'].mean().plot(figsize=(10,8),ax=ax2,color='b',alpha=.75);
ax1.set_ylabel('Total words');
ax2.set_ylabel('Lexical Density');
ax1.set_xlabel('')
ax2.set_xlabel('Year of release');
df_final[df_final.year==1984]
df_final.year.value_counts()[::-1].head(10)
group = df_final.groupby(['artist']).agg({"text_words":"mean"}).sort_values(by='text_words',ascending=False)[:10]
group.plot(kind='barh',figsize=(10,8),color='b',alpha=.75);
plt.ylabel('Artist', fontsize=16)
plt.xlabel('Length of review (words)', fontsize=16);
group = df_final.groupby(['artist']).agg({"lex_den":"mean"}).sort_values(by='lex_den',ascending=False)[:10]
group.plot(kind='barh',figsize=(10,8),color='b',alpha=.75);
plt.ylabel('Artist', fontsize=16)
plt.xlabel('Lexical density', fontsize=16);