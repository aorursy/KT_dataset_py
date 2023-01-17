# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/boardgamegeek2020/bgg_db.csv')
df.head()
df.shape
for col in df.columns:
    print(col)
df.dtypes
df.sort_values(by='rank', ascending=True, axis=0, inplace=True)
df.drop(['bgg_url', 'thumb_url', 'expands', 'reimplements', 'image_url'], axis=1, inplace=True)
df_year = df[(df.year >= 1980) & (df.year <= 2020)].groupby('year')
year_counts = df_year.year.count()
plt.figure(figsize=(14, 6))
plt.xticks(fontsize=14, rotation=90)
plt.yticks(fontsize=14)
sns.barplot(x=year_counts.index, y=year_counts)
plt.title('Games Published each Year', fontsize=18)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Games Published', fontsize=14)
plt.show()
# Cutting extreme cases out, because games without any rating are listed as 0 instead of NaN, and I wanted to exclude games with one 10.0 rating
df_year = df[(df.avg_rating >= 0.1) & (df.avg_rating <= 4.9) & (df.geek_rating >= 0.1) & (df.geek_rating <= 4.9) & (df.year >= 1980) & (df.year <= 2020)] 
line1 = df_year.groupby('year').avg_rating.mean()
line2 = df_year.groupby('year').geek_rating.mean()
plt.figure(figsize=(14, 6))
plt.title('Average Ratings by Year', fontsize=18)
plt.xlabel('Year')
plt.ylabel('Ratings')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
sns.lineplot(data=(line1, line2), markers=True, linewidth=3)
sns.set_style('darkgrid')
plt.show()
df_top300 = df[(df.year >= 1980) & (df.year <= 2020)][:300]
plt.figure(figsize=(14, 6))
sns.distplot(df_top300['year'], bins=40, color='darkblue')
plt.title('Top 300 Games by Year', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Percentage of Rank 300 Games', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
weight_year_narrowed = df[(df.weight >= 0.1) & (df.weight <= 4.9) & (df.year >= 1980) & (df.year <= 2020)] # Once again, cutting out extremes to get a more realistic picture
weight_year = weight_year_narrowed.groupby('year').weight.mean()
plt.figure(figsize=(14,6))
plt.title('Average Weight of Games by Year (0.1 - 4.9)', fontsize=18)
plt.xlabel('Year')
plt.ylabel('Average Weight')
sns.lineplot(data=weight_year, color='darkgreen', linewidth=3)
sns.set(style='darkgrid')
plt.show()
weight_year_narrowed = df[(df.weight >= 0.1) & (df.weight <= 4.9)]
bins = [-1, 1, 2, 3, 4, 5]
df['weight_cat'] = pd.cut(weight_year_narrowed['weight'], bins=bins, labels=bins[1:])
weight = [df[df['weight_cat'] == i]['avg_rating'] for i in range(1,6)]
plt.figure(figsize=(16, 8))
plt.title('Average Rating of Game Weight Groupings', fontsize=18)
plt.xlabel('Game Weight', fontsize=14)
plt.ylabel('Average Rating', fontsize=14)
ax = sns.boxplot(data=weight, palette='deep')
ax.set(xticklabels=['0-1', '1-2', '2-3', '3-4', '4-5'])
sns.set_style('whitegrid')
plt.show()
#Creating a new dataframe in order to count mechanics in the lists provided in df.mechanic

mechanics_lang = df.mechanic.str.split(',', expand=True) 
mechanics_lang = mechanics_lang.dropna(how='all')
mechanics_lang_num = mechanics_lang.fillna(0).apply(pd.Series.value_counts)
mechanics_lang_num = mechanics_lang_num[0].sort_values(ascending=False)

plt.figure(figsize=(14,6))
plt.title('Top 10 Most Mechanics in Games', fontsize=18)
plt.xlabel('Number of Games with Mechanic')
plt.ylabel('')
plt.yticks(fontsize=14)
mechanics_lang_num[:10].plot(kind='barh')
plt.gca().invert_yaxis()
plt.show()
# Creating a new dataframe in order to count Game Designers in lists provided in df.designer:

game_designer_lang = df.designer.str.split(',', expand=True)
game_designer_lang = game_designer_lang.dropna(how='all')
game_designer_lang_num = game_designer_lang.fillna(0).apply(pd.Series.value_counts)
game_designer_lang_num = game_designer_lang_num[0].sort_values(ascending=False).drop('(Uncredited)')

plt.figure(figsize=(10, 7))
plt.title('Most publishing Game Designer', fontsize=18)
plt.xlabel('Games Published', fontsize=14)
plt.yticks(fontsize=13)
game_designer_lang_num[:10].plot(kind='barh')
plt.gca().invert_yaxis()
plt.show()
concatenated = pd.concat([df, game_designer_lang], axis=1)
concatenated_grouped = concatenated[concatenated['num_votes'] > 2000].groupby(0).filter(lambda x: x[0].count() > 2)
game_designer_avg_rating = concatenated_grouped.groupby(0).avg_rating.mean().sort_values(ascending=False)
game_designer_geek_rating = concatenated_grouped.groupby(0).geek_rating.mean().sort_values(ascending=False)
game_designer_avg_rating = game_designer_avg_rating[:20]
game_designer_geek_rating = game_designer_geek_rating[:20]

plt.figure(figsize=(10, 10))
sns.barplot(x=game_designer_avg_rating, y=game_designer_avg_rating.index)
plt.title('Top 20 Best Game Designers by Average Rating', fontsize=18)
plt.xlabel('Average Ratings', fontsize=14)
plt.ylabel('Rating')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
plt.figure(figsize=(10, 10))
sns.barplot(x=game_designer_geek_rating, y=game_designer_geek_rating.index)
plt.title('Top 20 Best Game Designers by Geek Rating', fontsize=18)
plt.xlabel('Geek Ratings', fontsize=14)
plt.ylabel('Rating')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
game_designer_lang_top300 = game_designer_lang[:300].fillna(0).apply(pd.Series.value_counts)
game_designer_lang_top300 = game_designer_lang_top300[0].sort_values(ascending=False).drop('(Uncredited)')
plt.figure(figsize=(14,10))
plt.title('Game Designers most often in Top 300 Ranked Games', fontsize=18)
plt.xlabel('Games in Top 300 Ranked Games')
plt.ylabel('')
plt.yticks(fontsize=14)
game_designer_lang_top300[:20].plot(kind='barh')
plt.gca().invert_yaxis()
plt.show()