import numpy as np

import pandas as pd

import ast

import os

print(os.listdir("../input"))

import plotly.graph_objs as go

import plotly.plotly as py

import matplotlib.pyplot as plt

import seaborn as sns

import cufflinks

pd.options.display.max_columns = 30

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'

from plotly.offline import iplot

import json

from pandas.io.json import json_normalize

from wordcloud import WordCloud, STOPWORDS

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')
df = pd.read_csv('../input/ted_main.csv')

import datetime

month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

df['film_date'] = df['film_date'].apply(lambda x: datetime.datetime.fromtimestamp( int(x)).strftime('%d-%m-%Y'))

df['published_date'] = df['published_date'].apply(lambda x: datetime.datetime.fromtimestamp( int(x)).strftime('%d-%m-%Y'))

df = df[['tags', 'speaker_occupation', 'views', 'ratings', 'name', 'title', 'description', 'main_speaker',  'num_speaker', 'duration', 'event', 'film_date', 'published_date', 'comments', 'languages', 'related_talks', 'url']]

df['year'] = df['film_date'].apply(lambda x: x.split('-')[2]) # Add year column

df.iloc[1]['ratings']

df['ratings'] = df['ratings'].apply(lambda x: ast.literal_eval(x))

df.columns

df['humorous'] = df['ratings'].apply(lambda x: x[0]['count'])

df['courageous'] = df['ratings'].apply(lambda x: x[1]['count'])

df['persuasive'] = df['ratings'].apply(lambda x: x[-4]['count'])

df['jawdrop'] = df['ratings'].apply(lambda x: x[-3]['count'])

df['beautiful'] = df['ratings'].apply(lambda x: x[3]['count'])

df.head()
import ast

df['tags'] = df['tags'].apply(lambda x: ast.literal_eval(x))

s = df.apply(lambda x: pd.Series(x['tags']),axis=1).stack().reset_index(level=1, drop=True)

s.name = 'theme'

theme_df = df.drop('tags', axis=1).join(s)
theme_df.head(20)
year_df = pd.DataFrame(df['year'].value_counts().reset_index())

year_df.columns = ['year', 'talks']

plt.figure(figsize=(18,5))

sns.pointplot(x='year', y='talks', data=year_df)
occupation_df = df.groupby('speaker_occupation').count().reset_index()[['speaker_occupation', 'comments']]

occupation_df.columns = ['occupation', 'appearances']

occupation_df = occupation_df.sort_values('appearances', ascending=False)

fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(15, 8))

sns.boxplot(x='speaker_occupation', y='views', data=df[df['speaker_occupation'].isin(occupation_df.head(12)['occupation'])], palette="cool", ax =ax)

ax.set_ylim([0, 0.4e7])

plt.show()
pop_themes = pd.DataFrame(theme_df['theme'].value_counts()).reset_index()

pop_themes.columns = ['theme', 'talks']

pop_theme_talks = theme_df[(theme_df['theme'].isin(pop_themes.head(8)['theme'])) & (theme_df['theme'] != 'TEDx')]

pop_theme_talks['year'] = pop_theme_talks['year'].astype('int')

pop_theme_talks = pop_theme_talks[pop_theme_talks['year'] > 2008]
plt.figure(figsize=(15,8))

sns.barplot(x='theme', y='talks', data=pop_themes.head(10), palette = "cool")

plt.xlabel("Themes",size = 18)

plt.ylabel("Number of Talks", size = 18)

plt.xticks(size = 20, rotation = 30)

plt.yticks(size = 20)

plt.show();

plt.savefig('bargraph.png')
themes = list(pop_themes.head(8)['theme'])

themes.remove('TEDx')

ctab = pd.crosstab([pop_theme_talks['year']], pop_theme_talks['theme']).apply(lambda x: x/x.sum(), axis=1)
ctab[themes].plot(kind='line', stacked=False, colormap='rainbow', figsize=(13,8)).legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()

plt.savefig('linegraph.png')
df.to_csv('Hackathon.csv')
occupation_df.to_csv('occupation.csv')
theme_df.to_csv('themes.csv')
new_df = df[['title', 'main_speaker', 'views', 'published_date', 'beautiful', 'url']].sort_values('beautiful', ascending=False)[:10]

new_df.reset_index().to_json('beautiful.json')
new_df = df[['title', 'main_speaker', 'views', 'published_date', 'jawdrop', 'url']].sort_values('jawdrop', ascending=False)[:10]

new_df.reset_index().to_json('jawdrop.json')

new_df
new_df = df[['title', 'main_speaker', 'views', 'published_date', 'humorous', 'url']].sort_values('humorous', ascending=False)[:10]

new_df.reset_index().to_json('humorous.json')

new_df
new_df = df[['title', 'main_speaker', 'views', 'published_date', 'courageous', 'url']].sort_values('courageous', ascending=False)[:10]

new_df.reset_index().to_json('courageous.json')

new_df