# This Python 3 environment comes with many helpful analytics libraries installed



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import seaborn as sns

# Any results you write to the current directory are saved as output.
sns.set_palette('Set3', 10)

sns.set_context('talk')

movie=pd.read_csv('../input/Netflix Shows.csv',encoding='cp437')

movie.head()
movie.groupby(['rating']).size().plot(kind='bar')
movie=movie.fillna(value=0,axis=1)

movie=movie[movie['rating']!='UR']

movie.groupby('rating')['user rating score'].mean().sort_values()
movie.groupby(['release year']).size().plot(kind='bar')
movie=movie[movie['release year']>1940]

score=movie.groupby('rating')['rating']

score_counts=score.count()

movie_genre= movie[movie.rating.isin(score_counts.keys())]

table_score = pd.pivot_table(movie_genre ,values=['user rating score'],index=['release year'],columns=['rating'],aggfunc='mean',margins=False)

table_count=pd.pivot_table(movie_genre,values=['user rating score'],index=['release year'],columns=['rating'],aggfunc='count',margins=False)

plt.figure(figsize=(17,16))

sns.heatmap(table_score['user rating score'],linewidths=.5,annot=True,vmin=0,vmax=100,cmap='YlGnBu')

plt.title('Average scores of shows')
plt.figure(figsize=(17,16))

sns.heatmap(table_count['user rating score'],linewidths=1,annot=True,fmt='2.0f',vmin=0)

plt.title('Count of shows')
movie.groupby(['rating']).size().plot(kind='pie',autopct='%1.1f%%')
plt.figure(figsize=(15,8))

plt.xlim(1976,2017)

plt.ylim(0,100)

sns.kdeplot(movie['release year'], movie['user rating score'], n_levels=20, cmap="Reds", shade=True, shade_lowest=False)
table = movie.groupby('release year').size()

f,ax = plt.subplots(1,1,figsize=(17,10))

table.plot(ax=ax,c='red')


