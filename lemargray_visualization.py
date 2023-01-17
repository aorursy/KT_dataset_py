
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
games = pd.read_csv('../input/ign.csv', index_col=0)
games.head()
del games["url"]
games.head()
games = games.dropna(how='any',axis=0)
games['genre'].value_counts()[:10].plot(kind='pie',autopct='%1.1f%%',shadow=True,explode=[0.1,0,0,0,0,0,0,0,0,0])
plt.title('Distribution Of Top Genre"s')
fig=plt.gcf()
fig.set_size_inches(7,7)
plt.show()
fig,ax=plt.subplots(1,2,figsize=(22,10))
sns.countplot(games['release_month'],ax=ax[0],palette='Set1').set_title('Releases On Months')
plt.ylabel('')
sns.countplot(games['release_year'],ax=ax[1],palette='Set1').set_title('Releases on Years')
plt.xticks(rotation=90)
plt.show()
games['score'].mean()
games['score'].hist(edgecolor='black')
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.axvline(games['score'].mean(),color='b',linestyle='dashed')
games['platform'].value_counts()[:10].plot.pie(autopct='%1.1f%%',shadow=True,explode=[0.1,0,0,0,0,0,0,0,0,0])
fig=plt.gcf()
fig.set_size_inches(7,7)
plt.title('Top Platforms For Games')
plt.subplots(figsize=(15,15))
max_genres=games.groupby('genre')['genre'].count()
max_genres=max_genres[max_genres.values>200]
max_genres.sort_values(ascending=True,inplace=True)
mean_games=games[games['genre'].isin(max_genres.index)]
abc=mean_games.groupby(['release_year','genre'])['score'].mean().reset_index()
abc=abc.pivot('release_year','genre','score')
sns.heatmap(abc,annot=True,cmap='RdYlGn',linewidths=0.4)
plt.title('Average Score By Genre"s')
plt.show()
clone=games.copy()
clone['score_phrase']=clone['score_phrase'].map({'Awful':'Flop','Bad':'Flop','Disaster':'Flop','Unbearable':'Flop','Painful':'Flop','Mediocre':'Good','Okay':'Good','Great':'Hit','Amazing':'Hit','Masterpiece':'Masterpiece','Good':'Good'})
clone=clone[['score_phrase','platform','genre','score']]
max_platforms=clone['platform'].value_counts().index[:10]
plat=clone[clone['platform'].isin(max_platforms)]
plat=plat.groupby(['platform','score_phrase'])['score'].count().reset_index()
plat=plat.pivot('platform','score_phrase','score')
plat.plot.barh(width=0.9)
fig=plt.gcf()
fig.set_size_inches(12,14)
plt.show()
new_genres=max_genres.sort_values(ascending=False)[:10]
top_genres=games[games['genre'].isin(new_genres[:10].index)]
top_genres=top_genres.groupby(['release_year','genre'])['score'].count().reset_index()
top_genres=top_genres.pivot('release_year','genre','score')
sns.heatmap(top_genres,annot=True,fmt='2.0f',cmap='RdYlGn',linewidths=0.4)
fig=plt.gcf()
fig.set_size_inches(11,11)
plt.title('Releases By Top Genres By Years')
plt.show()

