# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



df=pd.read_csv('../input/albumlist.csv', encoding='latin1')



print(df.head())

print(df.info())

print(df.describe())



df.sort_values(by=['Year', 'Artist'])

df=df.reset_index()



# First let's see who's the all time king according to The Rolling Stone



print('\nAll time winner by now is:', df['Artist'].value_counts().idxmax(), 'with', df['Year'][df.Artist == 'Bob Dylan'].count(), 'best songs.')



# We check if there were more artists with ten best song prices.

print('\nThe best 15 are listed below:\n\n', df['Artist'].value_counts()[:15])

print('\nWe can see that it was wrong, The Beatles and The Rolling Stones also arrived at the first place.')



#There is an artist in the first 15: Various Artists, we simply remove it, cause it covers more than one performer.



df = df[df.Artist != 'Various Artists']



#There are cols that we do not need, make a new DataFrame with the needed ones.



df_grouped=df.groupby('Artist')

artists=df_grouped.size()

df_art=pd.DataFrame({'Artist': artists.index, 'Prizes': artists.values})



df_art=df_art.sort_values('Prizes', ascending=False)

df_art.index=np.arange(1, len(df_art)+1)

print(df_art.head())



print('\nThe best 15 are listed below:\n\n', df_art[:15])



# See it on a barplot



df_art=df_art[:15]



fig = plt.figure(figsize = (10, 10))

sns.set_style("whitegrid")

ax = sns.barplot(x='Artist', y='Prizes', data=df_art)



plt.title('Artists by the no. of wins', fontsize = 20)

plt.xlabel('Artist', fontsize = 15)

plt.setp(ax.get_xticklabels(), rotation=60)

plt.ylabel('Prizes', fontsize = 15)

plt.show()



#Explore the genres too.

genre_grouped=df.groupby('Genre')

genres=genre_grouped.size()

df_genre=pd.DataFrame({'Genre': genres.index, 'Prizes': genres.values})



df_genre=df_genre[:20]

df_genre=df_genre.sort_values('Prizes', ascending=False)



fig = plt.figure(figsize = (10, 10))

sns.set_style("whitegrid")

ax = sns.barplot(x='Genre', y='Prizes', data=df_genre)



plt.title('Genres by the no. of wins', fontsize = 20)

plt.xlabel('Genre', fontsize = 15)

plt.setp(ax.get_xticklabels(), rotation=90)

plt.ylabel('Prizes', fontsize = 15)

plt.show()



fig = plt.figure(figsize=(12, 12))



plt.pie(df_genre['Prizes'], autopct='%1.0f%%', pctdistance=0.85,)

plt.title('Most popular genres according to R.S. magazine')

plt.legend(df_genre['Genre'], loc='lower left')

plt.show()



#Let's put them together (genres and artists)

dropcols=['Number','Year','Album','Subgenre']

df_dropped=df.drop(dropcols, axis=1)

print(df_dropped.head())



df_dropped=df_dropped.groupby(['Artist', 'Genre']).size().sort_values(ascending=False).reset_index(name='Count')[:20]

print('\n', df_dropped.head())



sns.set_style("whitegrid")

plt.figure(figsize=(12,8))

ax = sns.barplot(x='Artist', y='Count',hue='Genre', data=df_dropped)

plt.legend()

plt.xticks(rotation=90)

plt.show()



plt.scatter(x=df_dropped['Artist'], y=df_dropped['Genre'], marker='*', color='red')

plt.xticks(rotation=90)

plt.show()
