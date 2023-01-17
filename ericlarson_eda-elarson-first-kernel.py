# My first kernel- exciting!!! 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling

import sqlite3 as sql

import datetime as dt
con=sql.connect("/kaggle/input/pitchfork-data/database.sqlite")

df=pd.read_sql_query("SELECT DISTINCT r.reviewid, r.title, r.artist, r.score, r.author, r.pub_date, r.pub_year, y.year as album_year, g.genre, l.label FROM reviews r LEFT OUTER JOIN years y on r.reviewid=y.reviewid LEFT OUTER JOIN genres g on r.reviewid=g.reviewid LEFT OUTER JOIN labels l on r.reviewid=l.reviewid", con)

df['album_year']=df['album_year'].astype('Int64')

df.head()

#Get some basic stats on the scores

dfscore=df[['reviewid','score']].drop_duplicates()

dfscore['score'].describe()
#Pitchfork does reviews of "significant albums from the past" weekly. I wanted to see whether these "late/retrospective" have a 

#different average score than albums that are reviewed at the time of release.

dfscore=df[['reviewid','score','album_year','pub_year']].drop_duplicates()

rsreviews=dfscore[dfscore['pub_year']-dfscore['album_year']>1]

treviews=dfscore[dfscore['pub_year']-dfscore['album_year']<=1]

print('Timely reviews have a mean score of {:.2f}.'.format(treviews['score'].mean()))

print('Retrospective reviews have a mean score of {:.2f}'.format(rsreviews['score'].mean()))
#Functions for quantiles

def q1(x):

    return x.quantile(0.25)

def q3(x):

    return x.quantile(0.75)



#Various metrics on scores by Genre

dfgenre=df[['reviewid','score','genre']].drop_duplicates()

dfgenre=dfgenre.groupby('genre').agg(mean_score=('score','mean'), 

                             median_score=('score','median'),

                             first_quant=('score',q1),

                             third_quant=('score',q3),

                             count=('score','count'))

print(dfgenre)

fig=plt.figure(figsize=(14,7))

ax=plt.gca()

genres=list(dfgenre.index)

positions=np.arange(len(genres))+1

heights=dfgenre['mean_score']

sns.barplot(genres,heights,palette='muted');

plt.ylim(6,8)

ax.tick_params(axis='both', which='both',length=0)

ax.set_title('Mean Score by Genre')

ax.set_ylabel('Mean Score');

#Let's find the average score by year. Note, average for all differs from above because an album can have multiple genres

dfyear=df[['reviewid','score','pub_year']].drop_duplicates()

print(dfyear.pivot_table('score','pub_year',aggfunc=['mean','median',q1,q3,'count'],margins=True))
#Let's find the average score by year and genre and plot it! Let's look from 2002 to 2016, since there is a similar # of reviews

import matplotlib.pyplot as plt

import seaborn as sns

dfgenrescore=df[(df['pub_year']>=2002)&(df['pub_year']<=2016)].drop_duplicates()

grouped=dfgenrescore.groupby(['pub_year','genre'])['score'].mean()

fig,ax=plt.subplots(figsize=(14,7))

grouped.unstack().plot(ax=ax)

ax.tick_params(axis='both', which='both',length=0)

ax.set_xlabel('Year of Review')

ax.set_ylabel('Mean Score')

ax.set_title('Average Score by Genre')

plt.legend(loc=(1.05,.5));

#Find the trend in musicians' albums scores over time (e.g. average first score, second score, third score,etc...). See if this 

#trend differs with the number of albums Pitchfork reviews. For example, artists who have only had one album reviewed by Pitchfork

#may have received lower scores on their first album than artists who had multiple albums reviewed by Pitchfork. I'm also curious

#if the scores of a given artist's albums decrease with time. Being hipsters (like me!), maybe Pitchfork gets bored of a particular

#artist/sound with time.



import seaborn as sns

dftrend=df[['reviewid','artist','score','album_year']].drop_duplicates()

dftrend=dftrend.sort_values(['album_year','reviewid']).dropna(subset=['album_year'])

#Adding the order column, which adds a sequential column (by album date) that indicates which # review this is for the artist

dftrend['num_album_reviewed']=dftrend.groupby('artist').cumcount()+1

#Removing 'various artists', since they have 694 albums and shouldn't count for this

dftrend=dftrend[dftrend['artist']!='various artists']

#Adding the max_order column, which indicates how many reviews an artist has

dftrend['tot_albums_reviewed']=dftrend.groupby('artist')['num_album_reviewed'].transform('count')

#For purposes of this analysis, let's make the maximum group of # of albums reviewed 5 or more and only look at albums up to

#the fifth album

dftrend['tot_albums_reviewed_group']=dftrend['tot_albums_reviewed']

dftrend=dftrend[dftrend['num_album_reviewed']<=5]

dftrend.loc[dftrend['tot_albums_reviewed_group']>=5,'tot_albums_reviewed_group']=5

#Setting up a new data frame, dftotal, to store the grouped means

dftotal=pd.DataFrame()

groups=dftrend['tot_albums_reviewed_group'].unique()

#Let's find the average score for each album reviewed by the order in which it was released, grouped by the # of albums

#Pitchfork reviewed by the artist

for i in groups:

    dftemp=dftrend[dftrend['tot_albums_reviewed_group']==i]

    dftemp=dftemp.groupby(['tot_albums_reviewed_group','num_album_reviewed'])['score'].agg(['mean','median',q1,q3,'count'])

    dftemp=dftemp.reset_index()

    dftotal=dftotal.append(dftemp)

dftotal=dftotal.sort_values(['tot_albums_reviewed_group','num_album_reviewed'])

dftotal=dftotal.reset_index(drop=True)

print(dftotal)



#Regardless of how many albums Pitchfork has reviewed for an artist, it looks like the average score decreases with each album.

#In addition, the more albums Pitchfork has reviewed for a particular artist, the higher scores the respective album receives. 

#For example, for artists with one album reviewed on Pitchfork, the average score of that album is 6.79. For artists with 5 

#or more albums reviewed on Pitchfork, the average score of that first album is 7.61.
#Let's graph this wonderful data!

import matplotlib.ticker as ticker



dftotal['tot_albums_reviewed_group']=dftotal['tot_albums_reviewed_group'].astype('str')

sns.lineplot(x='num_album_reviewed',y='mean',hue='tot_albums_reviewed_group',style='tot_albums_reviewed_group',

             palette=['red','lightcoral','maroon','dimgrey','black'],markers=True,data=dftotal);

ax = plt.gca()

ax.set_xlabel('# Album Released by Artist')

ax.set_ylabel('Mean Score')

ax.set_title('Average Review Score by # of Album Released\nand Total Albums Reviewed by Pitchfork')

ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

ax.tick_params(axis='both', which='both',length=0)

L=plt.legend()

L.get_texts()[0].set_text('Total Number of Albums\n Reviewed for Artist')

L.get_texts()[5].set_text('5 or more')



#Plot shows that the average score goes down as an artist releases more albums. In addition, the review for a given album

#(first, second, third, etc) tends to be higher if Pitchfork has reviewed more of the artists' albums. This may indicate that

#Pitchfork only continues to review albums of a certain score or that the artists who continue to make music receive higher

#scores on Pitchfork

#Let's find the trend between month published and average score (practice with datetimes!)

dfdates=df[['reviewid','score','pub_date']].drop_duplicates()

dfdates.head(5)

dfdates['pub_date']=pd.to_datetime(dfdates['pub_date'],format='%Y-%m-%d')

dfdates['month']=dfdates['pub_date'].dt.month

dfdates['month_name']=dfdates['pub_date'].dt.month_name()

months=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

dfdates['month_name']=pd.Categorical(dfdates['month_name'], categories=months, ordered=True)

dfdatesagg=dfdates.groupby('month_name')['score'].agg(['mean','median',q1,q3,'count'])

print(dfdatesagg)



#Let's make a line graph of the scores!

fig=plt.figure()

ax=plt.gca()

plt.plot(list(dfdatesagg.index),dfdatesagg['mean'],'-o')

plt.xticks(rotation=45)

ax.tick_params(axis='both', which='both',length=0)

plt.ylim(6,8)

ax.set_ylabel('Mean Score')

ax.set_title('Mean Score by Review Month');
#Let's look at the length of the title and relationship to score

dftitle=df[['reviewid','title','score']].drop_duplicates()

dftitle['title_length']=dftitle['title'].map(lambda x: len(x.split()))

dftitle=dftitle[dftitle['title_length']!=0]

dftitle['title_length_11']=dftitle['title_length']

#Any title length greater than or equal to 11 words we'll categorize as 11

dftitle.loc[dftitle['title_length_11']>=11,'title_length_11']=11

dftitlegroup=dftitle.groupby('title_length_11')['score'].agg(['mean','median',q1,q3,'count'])

print(dftitlegroup)
title_lengths=list(dftitlegroup.index)

sns.lineplot(x=title_lengths,y='mean',data=dftitlegroup)

ax=plt.gca()

ax.set_title('Album Title Length and Mean Score');

ax.set_ylabel('Mean Score')

ax.set_xlabel('How Many Words in Album Title')

ax.tick_params(axis='both', which='both',length=0)

plt.xticks(np.arange(min(title_lengths), max(title_lengths)+1, 1.0))

plt.ylim(6,8);
sns.boxplot(x='title_length_11',y='score',data=dftitle)

ax=plt.gca()

ax.tick_params(axis='both', which='both',length=0)

ax.set_title('Album Title Length and Mean Score')

ax.set_ylabel('Mean Score')

ax.set_xlabel('How Many Words in Album Title');
