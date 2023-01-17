# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import re



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')

df.head(5)
df['release_year'] = pd.to_datetime(df['release_year'],format='%Y')

df['date_added'] = pd.to_datetime(df['date_added'])

indeks = df[np.isnan(df['date_added'])].index

df.loc[indeks, 'date_added'] = df.loc[indeks, 'release_year']

df['release_year'] = df['release_year'].dt.year
plt.figure(figsize=(6,8))

total = df['type'].value_counts();

sns.barplot(x=total.index, y=total.values)

plt.box(on=None)
release = df['release_year'].value_counts()

release = release.sort_index(ascending=True)



plt.figure(figsize=(8,6))

plt.plot(release[-11:-1])

plt.scatter(release[-11:-1].index, release[-11:-1].values, s=0.5*release[-11:-1].values, c='orange');

plt.box(on=None);

plt.xticks(release[-11:-1].index);

plt.title('Number of Programmes Released by Year', color='red', fontsize=15);
rating = df['rating'].value_counts()

#rating = rating.sort_values()



plt.figure(figsize=(8,6));

plt.title('Number of Programmes by Rating', color='red', fontsize=15)

#plt.barh(rating.index, rating.values, align='center');

sns.barplot(x=rating.values, y=rating.index, palette='gnuplot');

plt.box(on=None);

plt.xlabel('Number of Programmes');
country_rating = df.groupby(['country', 'rating']).count().sort_values('show_id', ascending=False)



plt.figure(figsize=(8,6))

sns.barplot(y=country_rating.index[:10], x = country_rating['show_id'][:10], palette='gnuplot2');

plt.box(on=None)

plt.title('Most Popular Programmes by Country & Rating', fontsize=15, color='red')

plt.xlabel('Number of Programmes');
movie = df.copy()

movie = movie[movie['type'] == 'Movie']



movie['minute'] = [int(re.findall('\d{1,3}', w)[0]) for w in movie.duration.ravel()]



duration_rating = movie.groupby(['rating']).mean().sort_values('minute')



plt.figure(figsize=(8,6))

sns.barplot(x=duration_rating.index, y=duration_rating.minute.values, palette='gnuplot_r')

plt.box(on=None)

plt.title('Number of Movies by Rating', fontsize=15, color='red');

plt.xlabel('Movie Rating');
duration_year = movie.groupby(['release_year']).mean().sort_values('minute')

duration_year = duration_year.sort_index()



plt.figure(figsize=(15,6))

sns.lineplot(x=duration_year.index, y=duration_year.minute.values)

plt.box(on=None)

plt.ylabel('Movie duration in minutes');

plt.xlabel('Year of released');

plt.title("YoY Trends of Movie's Duration", fontsize=15, color='red');
plt.figure(figsize=(8,8))

sns.barplot(y=movie.director.value_counts()[:10].sort_values().index, x=movie.director.value_counts()[:10].sort_values().values);

plt.title('Most Productive Movie Director', color='red', fontsize=15)

plt.box(on=None)

plt.xticks(movie.director.value_counts()[:10].sort_values().values);

plt.xlabel('Number of Movies Released');
director_minute = movie.groupby('director').sum().sort_values('minute', ascending=False)

plt.figure(figsize=(8,8))

sns.barplot(y=director_minute.index[:10], x=director_minute.minute[:10]);

plt.title('Most Productive Movie Director in Video Length', color='red', fontsize=15)

plt.box(on=None)

plt.xlabel('Length of Movies Released');