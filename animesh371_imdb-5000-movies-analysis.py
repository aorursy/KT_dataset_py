import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import matplotlib

matplotlib.style.use('ggplot')

%pylab inline
df = pd.read_csv('../input/movie_metadata.csv')
df.head()
df.dtypes
df.shape
df[['gross','duration','imdb_score']].dropna(axis=0, how='any').describe()
df[['gross','duration','imdb_score']].dropna(axis=0, how='any').plot.scatter(x='gross', y='imdb_score')
plt.show()
df[['gross','duration','imdb_score']].dropna(axis=0, how='any').plot.scatter(x='duration', y='imdb_score')
plt.show()
df['duration'].describe()
gross_corr = df['imdb_score'].corr(df['gross'])

duration_corr = df['imdb_score'].corr(df['duration'])
print ("Correlation between gross and imdb score " + str(gross_corr))
print ("Correlation between duration and imdb score " + str(duration_corr))
fb_like_names = ['director_facebook_likes', 'actor_2_facebook_likes', 'actor_1_facebook_likes', 'actor_3_facebook_likes', 'imdb_score'

]



df[fb_like_names].head()
df[fb_like_names].dropna(axis=0, how='any').plot.scatter(x='director_facebook_likes', y='imdb_score')
plt.show()
df[fb_like_names].dropna(axis=0, how='any').plot.scatter(x='actor_1_facebook_likes', y='imdb_score')

plt.show()
df[fb_like_names].dropna(axis=0, how='any').plot.scatter(x='actor_2_facebook_likes', y='imdb_score')

plt.show()
df[fb_like_names].dropna(axis=0, how='any').plot.scatter(x='actor_3_facebook_likes', y='imdb_score')

plt.show()
director_corr = df['imdb_score'].corr(df['director_facebook_likes'])

actor_3_corr = df['imdb_score'].corr(df['actor_3_facebook_likes'])

actor_2_corr = df['imdb_score'].corr(df['actor_2_facebook_likes'])

actor_1_corr = df['imdb_score'].corr(df['actor_1_facebook_likes'])
print ("Director correlation = " + str(director_corr) + " Actor_1 correlation - " + str(actor_1_corr) + " Actor_2 correlation - " + str(actor_2_corr) + " Actor_3 correlation - " + str(actor_3_corr))
df['country'].shape
country = df['country'].dropna(axis=0, how='any').value_counts()

country_filtered = country[country > 10]
country_filtered.plot(kind='bar', color=['g'])

plt.show()
title_year = df['title_year'].dropna(axis=0, how='any').value_counts()
title_year.plot(kind='bar', color=['r'])

plt.show()
pylab.rcParams['figure.figsize'] = (20, 6)

title_year.sort_index()[title_year.sort_index()>10].plot(kind='bar', color=['r'])
title_year.sort_index().describe()
df.dtypes
df.isnull().sum()
df['color'].isnull().sum()
df['color'].value_counts()
df['color'].shape
df[df['color']==' Black and White'].head()
blacknwhite_pic = df[df['color']==' Black and White']

colored_pic = df[df['color']=='Color']

colored_pic.head()
colored_pic['title_year'].value_counts().head()
blacknwhite_pic['title_year'].value_counts().head()
colored_pic['title_year'].value_counts()[colored_pic['title_year'].value_counts() > 10].sort_index().plot(kind='bar', color=['g'])
colored_pic['title_year'].value_counts()[colored_pic['title_year'].value_counts() > 15].describe()
blacknwhite_pic['title_year'].value_counts()[blacknwhite_pic['title_year'].value_counts() > 1].sort_index().plot(kind='bar', color=['y'])
df['color'] = df['color'].fillna('Color')
df['director_name'].value_counts()[df['director_name'].value_counts() > 10].plot(kind='bar', color='g')
df['director_name'].value_counts().head()
df[df['director_name']=='Steven Spielberg'].head()
df[['director_name', 'director_facebook_likes']].drop_duplicates().sort_values('director_facebook_likes', ascending='False').tail(20)