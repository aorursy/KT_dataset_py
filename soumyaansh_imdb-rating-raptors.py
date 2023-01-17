import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

sns.set_context("notebook", font_scale=1.1)

sns.set_style("ticks")
df = pd.read_csv('../input/movie_metadata.csv')

df.head()
print(df.isnull().sum())
# content_rating has 303 null values

df = df.dropna(subset=['content_rating'])
df['imdb_score'] = df['imdb_score'].round()

df['cast_total_facebook_likes'].sort_values(ascending=False).head(10)
df = df[(df.cast_total_facebook_likes != 656730) & (df.cast_total_facebook_likes != 303717) & (df.cast_total_facebook_likes != 263584)]
fig = plt.figure(figsize=(13, 5))

t = np.arange(0.01, 20.0, 0.01)

g = sns.lmplot('cast_total_facebook_likes', 'imdb_score',data=df,fit_reg=False,hue='content_rating',palette='muted',x_jitter=2.0,y_jitter=2.0,size=10)

g.set(xlim=(0, None))

g.set(ylim=(0, None))

x_values = [10000,30000, 50000, 70000, 90000, 110000, 130000, 150000, 170000, 190000]

plt.xticks(x_values)

plt.title('Impact of Facebook likes on IMDb score')

plt.ylabel('IMDB Rating')

plt.xlabel('Total Facebook Likes')

plt.show()