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

from textblob import TextBlob

import warnings
warnings.filterwarnings('ignore')

import time
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
df_final.drop_duplicates('reviewid',inplace=True)
df_final['text_len'] = df_final['content'].apply(lambda x: len(x))
df_final['text_words'] = df_final['content'].apply(lambda x: len(x.split()))
df_final['uniq_words'] = df_final['content'].apply(lambda x: len(set(x.split())))
df_final['lex_den'] = 100*df_final['uniq_words']/df_final['text_words']
df_final.reset_index(inplace=True,drop=True)
df_final.head()
def polarity_calc(text):
    
    blob = TextBlob(text)
    pol_list = []
    for sent in blob.sentences:
        pol_list.append(sent.sentiment.polarity)
        
    return np.sum(pol_list)
def subjectivity_calc(text):
    
    blob = TextBlob(text)
    sub_list = []
    for sent in blob.sentences:
        sub_list.append(sent.sentiment.subjectivity)
        
    return np.sum(sub_list)
start = time.time()
df_final['polarity'] = df_final['content'].apply(polarity_calc)
end = time.time()
print("Time elapsed:: ",end-start,"(s)")
start = time.time()
df_final['subjectivity'] = df_final['content'].apply(subjectivity_calc)
end = time.time()
print("Time elapsed:: ",end-start,"(s)")
df_final.head()
plt.figure(figsize=(10,8))
plt.scatter(df_final['polarity'],df_final['subjectivity'], alpha=.25, color='b');
plt.xlabel('Polarity')
plt.ylabel("Subjectivity");
df_final['polarity'].corr(df_final['subjectivity'])
df_final.groupby('genre')['polarity'].mean().plot(kind='barh', figsize=(10,8));
plt.xlabel('Polarity')
plt.ylabel('Genre');
df_final.groupby('genre')['subjectivity'].mean().plot(kind='barh', figsize=(10,8));
plt.xlabel('Subjectivity')
plt.ylabel('Genre');
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8,6))
df_final.groupby(['year'])['polarity'].mean().plot(figsize=(10,8),ax=ax1);
df_final.groupby(['year'])['subjectivity'].mean().plot(figsize=(10,8),ax=ax2);
ax1.set_xlabel('Year');
ax2.set_xlabel('Year');
ax1.set_ylabel('Polarity')
ax2.set_ylabel('Subjectivity');
# Do overall score trends change by genre over the years?
df_final[df_final['genre']=='rock'].groupby(['year'])['polarity'].mean()\
.plot(figsize=(15,10),color='b',alpha=0.75, label='Rock')
df_final[df_final['genre']=='electronic'].groupby(['year'])['polarity'].mean()\
.plot(figsize=(15,10),color='r',alpha=0.75, label='Electronic')
df_final[df_final['genre']=='folk/country'].groupby(['year'])['polarity'].mean()\
.plot(figsize=(15,10),color='g',alpha=0.75, label='Folk/Country')
df_final[df_final['genre']=='pop/r&b'].groupby(['year'])['polarity'].mean()\
.plot(figsize=(15,10),color='y',alpha=0.75, label='Pop/RnB')
df_final[df_final['genre']=='jazz'].groupby(['year'])['polarity'].mean()\
.plot(figsize=(15,10),color='c',alpha=0.75, label='Jazz')
df_final[df_final['genre']=='rap'].groupby(['year'])['polarity'].mean()\
.plot(figsize=(15,10),color='m',alpha=0.75, label='Rap')
df_final[df_final['genre']=='experimental'].groupby(['year'])['polarity'].mean()\
.plot(figsize=(15,10),color='k',alpha=0.75, label='Experimental')
df_final[df_final['genre']=='metal'].groupby(['year'])['polarity'].mean()\
.plot(figsize=(15,10),color='gold',alpha=0.75, label='Metal')
df_final[df_final['genre']=='global'].groupby(['year'])['polarity'].mean()\
.plot(figsize=(15,10),color='plum',alpha=0.75, label='Global')
plt.xlabel('Year',fontsize=16)
plt.ylabel('Mean Polarity (of Sentiment)',fontsize=16);
plt.legend(fontsize=16);
# Do overall score trends change by genre over the years?
df_final[df_final['genre']=='rock'].groupby(['year'])['subjectivity'].mean()\
.plot(figsize=(15,10),color='b',alpha=0.75, label='Rock')
df_final[df_final['genre']=='electronic'].groupby(['year'])['subjectivity'].mean()\
.plot(figsize=(15,10),color='r',alpha=0.75, label='Electronic')
df_final[df_final['genre']=='folk/country'].groupby(['year'])['subjectivity'].mean()\
.plot(figsize=(15,10),color='g',alpha=0.75, label='Folk/Country')
df_final[df_final['genre']=='pop/r&b'].groupby(['year'])['subjectivity'].mean()\
.plot(figsize=(15,10),color='y',alpha=0.75, label='Pop/RnB')
df_final[df_final['genre']=='jazz'].groupby(['year'])['subjectivity'].mean()\
.plot(figsize=(15,10),color='c',alpha=0.75, label='Jazz')
df_final[df_final['genre']=='rap'].groupby(['year'])['subjectivity'].mean()\
.plot(figsize=(15,10),color='m',alpha=0.75, label='Rap')
df_final[df_final['genre']=='experimental'].groupby(['year'])['subjectivity'].mean()\
.plot(figsize=(15,10),color='k',alpha=0.75, label='Experimental')
df_final[df_final['genre']=='metal'].groupby(['year'])['subjectivity'].mean()\
.plot(figsize=(15,10),color='gold',alpha=0.75, label='Metal')
df_final[df_final['genre']=='global'].groupby(['year'])['subjectivity'].mean()\
.plot(figsize=(15,10),color='plum',alpha=0.75, label='Global')
plt.xlabel('Year',fontsize=16)
plt.ylabel('Mean Subjectivity',fontsize=16);
plt.legend(fontsize=16);
df_final[df_final['genre']=='rock']['polarity']\
.plot(kind='kde',figsize=(10,8),color='b',alpha=0.5, label='Rock')
df_final[df_final['genre']=='electronic']['polarity']\
.plot(kind='kde',figsize=(10,8),color='r',alpha=0.5, label='Electronic')
df_final[df_final['genre']=='folk/country']['polarity']\
.plot(kind='kde',figsize=(10,8),color='g',alpha=0.5, label='Folk/Country')
df_final[df_final['genre']=='pop/r&b']['polarity']\
.plot(kind='kde',figsize=(10,8),color='y',alpha=0.5, label='Pop/R&B')
df_final[df_final['genre']=='jazz']['polarity']\
.plot(kind='kde',figsize=(10,8),color='c',alpha=0.5, label='Jazz')
df_final[df_final['genre']=='rap']['polarity']\
.plot(kind='kde',figsize=(10,8),color='m',alpha=0.5, label='Rap')
df_final[df_final['genre']=='rap']['polarity']\
.plot(kind='kde',figsize=(10,8),color='m',alpha=0.5, label='Rap')
df_final[df_final['genre']=='experimental']['polarity']\
.plot(kind='kde',figsize=(10,8),color='k',alpha=0.5, label='Experimental')
df_final[df_final['genre']=='metal']['polarity']\
.plot(kind='kde',figsize=(10,8),color='gold',alpha=0.5, label='Metal')
df_final[df_final['genre']=='global']['polarity']\
.plot(kind='kde',figsize=(10,8),color='plum',alpha=0.5, label='Global')
plt.xlabel('Polarity',fontsize=16)
plt.ylabel('Density',fontsize=16);
plt.legend(fontsize=16);
df_final[df_final['genre']=='rock']['subjectivity']\
.plot(kind='kde',figsize=(10,8),color='b',alpha=0.5, label='Rock')
df_final[df_final['genre']=='electronic']['subjectivity']\
.plot(kind='kde',figsize=(10,8),color='r',alpha=0.5, label='Electronic')
df_final[df_final['genre']=='folk/country']['subjectivity']\
.plot(kind='kde',figsize=(10,8),color='g',alpha=0.5, label='Folk/Country')
df_final[df_final['genre']=='pop/r&b']['subjectivity']\
.plot(kind='kde',figsize=(10,8),color='y',alpha=0.5, label='Pop/R&B')
df_final[df_final['genre']=='jazz']['subjectivity']\
.plot(kind='kde',figsize=(10,8),color='c',alpha=0.5, label='Jazz')
df_final[df_final['genre']=='rap']['subjectivity']\
.plot(kind='kde',figsize=(10,8),color='m',alpha=0.5, label='Rap')
df_final[df_final['genre']=='rap']['subjectivity']\
.plot(kind='kde',figsize=(10,8),color='m',alpha=0.5, label='Rap')
df_final[df_final['genre']=='experimental']['subjectivity']\
.plot(kind='kde',figsize=(10,8),color='k',alpha=0.5, label='Experimental')
df_final[df_final['genre']=='metal']['subjectivity']\
.plot(kind='kde',figsize=(10,8),color='gold',alpha=0.5, label='Metal')
df_final[df_final['genre']=='global']['subjectivity']\
.plot(kind='kde',figsize=(10,8),color='plum',alpha=0.5, label='Global')
plt.xlabel('Subjectivity',fontsize=16)
plt.ylabel('Density',fontsize=16);
plt.legend(fontsize=16);
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8,6))
df_final.groupby(['author'])['polarity'].mean().sort_values(ascending=False)[:10].plot(kind='barh',figsize=(10,8),ax=ax1);
df_final.groupby(['author'])['subjectivity'].mean().sort_values(ascending=False)[:10].plot(kind='barh',figsize=(10,8),ax=ax2);
ax1.set_xlabel('Polarity');
ax2.set_xlabel('Subjectivity');
ax1.set_ylabel('')
ax2.set_ylabel('');
df_final[df_final.author=='simon goddard']
def positive_polarity_calc(text):
    
    blob = TextBlob(text)
    pos_pol_list = []
    for sent in blob.sentences:
        if sent.sentiment.polarity>0:
            pos_pol_list.append(sent.sentiment.polarity)
        
    return np.sum(pos_pol_list)

def negative_polarity_calc(text):
    
    blob = TextBlob(text)
    neg_pol_list = []
    for sent in blob.sentences:
        if sent.sentiment.polarity<0:
            neg_pol_list.append(sent.sentiment.polarity)
        
    return np.sum(neg_pol_list)
start = time.time()
df_final['positive_polarity'] = df_final['content'].apply(positive_polarity_calc)
df_final['negative_polarity'] = df_final['content'].apply(negative_polarity_calc)
end = time.time()
print("Time elapsed: ",end-start,"(s)")
df_final['pol_ratio'] = np.abs(df_final['positive_polarity']/df_final['negative_polarity'])
df_final.head()
df_final.corr()
# 1. Net Polarity
print("Net polarity, correlation with score")
for genre in df_final['genre'].unique():
    print(genre, df_final[df_final['genre']==genre]['score'].corr(df_final[df_final['genre']==genre]['polarity']))
# 2. Net positive polarity
print("Net positive polarity, correlation with score")
for genre in df_final['genre'].unique():
    print(genre, df_final[df_final['genre']==genre]['score'].corr(df_final[df_final['genre']==genre]['positive_polarity']))
# 2. Net negative polarity
print("Net negative polarity, correlation with score")
for genre in df_final['genre'].unique():
    print(genre, df_final[df_final['genre']==genre]['score'].corr(df_final[df_final['genre']==genre]['negative_polarity']))
