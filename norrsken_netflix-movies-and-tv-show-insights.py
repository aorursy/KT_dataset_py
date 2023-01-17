# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Loading the csv

df = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')

df.head()
# columns in the dataset

print(df.columns)

# shape of dataset

print(df.shape)
plt.figure(figsize = (15, 5))

sns.countplot(df['release_year'][df.release_year>2004], palette = 'PuBuGn')

df.release_year.value_counts()[:15]
# Grouping the titles by their release year and then by the type of content(ie. TV Show or Movies)

counts = df.groupby(['release_year', 'type'])['show_id'].count().to_frame().reset_index()

#splitting the type column into two columns (TV Show and Movie) which will contain the count of TV Shows and Movies releasedrespectively 

counts = pd.pivot_table(counts, values='show_id', index=['release_year'], columns=['type']).fillna(0).reset_index()



plt.figure(figsize = (20,5))

ax = sns.lineplot(x='release_year', y='Movie', data=counts, label = 'Movie')

ax2 = sns.lineplot(x='release_year', y='TV Show', data=counts, label = 'TV Show').set_title('Content count from each year')

plt.legend()

counts
plt.figure(figsize = (15,5))

ax = sns.lineplot(x='release_year', y='Movie', data=counts[40:])

ax2 = sns.lineplot(x='release_year', y='TV Show', data=counts[40:]).set_title('Content from last 40 years')
plt.figure(figsize = (15,5))

sns.countplot(df.rating, palette = 'plasma')
df['year_added'] = df['date_added'].fillna(df['release_year'])

df['year_added'] = df['year_added'].astype(str).apply(lambda x: x[-4:])



plt.figure(figsize = (15,5))

sns.countplot(df.year_added, palette = "ch:2.5,-.2,dark=.3")
plt.figure(figsize = (15,5))

sns.countplot(x='year_added', hue='type',data=df)
h = df['listed_in'].unique()

pd.set_option('display.max_rows', 500)

h[0].split(',')
# splitting the 'listed_in' column.

genres = df['listed_in'].str.split(',', 4, expand=True)

# pd.set_option('display.max_rows', genres.shape[0] + 1)



# adding title column and reordering the columns

genres['title'] = df['title']

genres = genres[['title', 0, 1, 2]]



#melting the genres columns into a single column.

genres = pd.melt(genres, id_vars=['title'], value_vars=[0, 1, 2])

genres = genres.dropna()

genres = genres.drop('variable', axis = 1)



# plotting the countplot for genres.

plt.figure(figsize=(25,10))

plot = sns.countplot(x = 'value', data = genres)

plot.set_xticklabels(plot.get_xticklabels(), rotation=90)

plt.xlabel('Genres')

plt.ylabel('Number of Titles')