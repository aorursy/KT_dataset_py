# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import plotly.express as px
df = pd.read_csv('/kaggle/input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv')
df.drop(['Unnamed: 0', 'ID'], axis=1, inplace=True)
df['Directors'] = df['Directors'].str.split(',')
df['Genres'] = df['Genres'].str.split(',')
df['Country'] = df['Country'].str.split(',')
df['Language'] = df['Language'].str.split(',')
df['Rotten Tomatoes'] = df['Rotten Tomatoes'][df['Rotten Tomatoes'].notnull()].str.replace('%', '').astype(float)
df
df.info()
plt.figure(figsize=(20, 10))
ax = sns.countplot('Year', data=df)
ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=90)
plt.show()
plt.figure(figsize=(15, 10))
ax = sns.regplot('Year', 'Runtime', data=df)
plt.figure(figsize=(15, 10))
ax = sns.distplot(df['IMDb'],kde=False)
plt.figure(figsize=(15, 10))
ax = sns.distplot(df['Rotten Tomatoes'], kde=False)
plt.figure(figsize=(15, 10))
ax = sns.regplot('Year', 'IMDb', data=df)
plt.figure(figsize=(15, 10))
ax = sns.regplot('Year', 'Rotten Tomatoes', data=df)
plt.figure(figsize=(15, 10))
ax = sns.regplot('Runtime', 'IMDb', data=df)
plt.figure(figsize=(15, 10))
ax = sns.regplot('Runtime', 'Rotten Tomatoes', data=df)
country_column = 'Country'
country_df = df[[country_column]].explode(country_column).groupby(country_column).size().to_frame(name='No of Movies')
country_df
fig = px.pie(
    country_df,
    values='No of Movies',
    names=country_df.index
)
fig.show()
temp_group = df[['Country', 'Genres']].explode('Country').explode('Genres').groupby('Country')
us_df = temp_group.get_group('United States').groupby('Genres').size().to_frame(name='No of Genres')

fig = px.pie(
    us_df,
    values='No of Genres',
    names=us_df.index
)
fig.show()
uk_df = temp_group.get_group('United Kingdom').groupby('Genres').size().to_frame(name='No of Genres')

fig = px.pie(
    uk_df,
    values='No of Genres',
    names=uk_df.index
)
fig.show()
india_df = temp_group.get_group('India').groupby('Genres').size().to_frame(name='No of Genres')

fig = px.pie(
    india_df,
    values='No of Genres',
    names=india_df.index
)
fig.show()
language_column = 'Language'
language_df = df[[language_column]].explode(language_column).groupby(language_column).size().to_frame(name='No of Movies')

fig = px.pie(
    language_df,
    values='No of Movies',
    names=language_df.index
)
fig.show()
genres_column = 'Genres'
genres_df = df[[genres_column]].explode(genres_column).groupby(genres_column).size().to_frame(name='No of Movies')

fig = px.pie(
    genres_df,
    values='No of Movies',
    names=genres_df.index
)
fig.show()
age_column = 'Age'
age_df = df[[age_column]].explode(age_column).groupby(age_column).size().to_frame(name='No of Movies')

fig = px.pie(
    age_df,
    values='No of Movies',
    names=age_df.index
)
fig.show()