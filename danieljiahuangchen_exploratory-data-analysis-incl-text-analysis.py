import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



df = pd.read_csv("../input/data.csv")

df = df.drop("Unnamed: 0", axis="columns")

df.head()
fig1 = plt.figure(figsize=(18, 12))



ax1 = fig1.add_subplot(331)

sns.countplot(x='key',hue='target',data=df, palette='BuGn')



ax2 = fig1.add_subplot(332)

sns.countplot(x='mode',hue='target',data=df, palette='BuGn')



ax3 = fig1.add_subplot(333)

sns.countplot(x='time_signature',hue='target',data=df, palette='BuGn')
sns.pairplot(df)
fig2 = plt.figure(figsize=(16, 8))

sns.heatmap(df.corr(), annot=True, annot_kws={'weight':'bold'},linewidths=.5, cmap='YlGnBu')
sns.lmplot(y='loudness',x='energy',data=df, hue='target',palette='BuGn')
sns.lmplot(y='energy',x='acousticness',data=df, hue='target',palette='BuGn')
import wordcloud

from subprocess import check_output

%pylab inline



songs_like = ' '.join(df[df['target']==1]['song_title']).lower().replace(' ',' ')

cloud_like = wordcloud.WordCloud(background_color='white',

                            mask=imread('Spotify.jpg'),

                            max_font_size=100,

                            width=2000,

                            height=2000,

                            max_words=1000,

                            relative_scaling=.5).generate(songs_like)



songs_dislike = ' '.join(df[df['target']==0]['song_title']).lower().replace(' ',' ')

cloud_dislike = wordcloud.WordCloud(background_color='white',

                            mask=imread('Spotify.jpg'),

                            max_font_size=100,

                            width=2000,

                            height=2000,

                            max_words=1000,

                            relative_scaling=.5).generate(songs_dislike)

fig3=plt.figure(figsize=(12,12))



ax4 = fig3.add_subplot(121)

plt.imshow(cloud_like)

plt.axis('off')

plt.title('Like (target=1)', fontsize=20, color='g', fontweight='bold')



ax5 = fig3.add_subplot(122)

plt.imshow(cloud_dislike)

plt.axis('off')

plt.title('Dislike (target=0)', fontsize=20, color='g', fontweight='bold')
artist_like = df[df['target']==1].groupby('artist').count().reset_index()[['artist', 'target']]

artist_like.columns = ['artist', 'appearances']

artist_like = artist_like.sort_values('appearances', ascending=False)

artist_like=artist_like.head(10)

plt.barh(left=0, y='artist', width='appearances', data=artist_like, color='g', alpha=0.7)

plt.title('Top 10 Favorite Artists', color='g', fontsize='xx-large')