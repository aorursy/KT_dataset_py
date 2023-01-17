# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

plt.style.use('seaborn-whitegrid')

sns.set_style('whitegrid')
missing_values=['n/a','na','--']

netflix=pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv', na_values=missing_values)

netflix.head()
def data_inv(df):

    print('Netflix Movies and Shows: ', df.shape[0])

    print('Database Variables: ', df.shape[1])

    print('-' * 30)

    print('Dataset Columns: \n')

    print(df.columns)

    print('-' * 30)

    print('Datatype of Each Column: \n')

    print(df.dtypes)

    print('-' * 30)

    print('Missing Rows in Each Column: \n')

    c=df.isnull().sum()

    print(c[c > 0])

data_inv(netflix)
netflix = netflix.drop('show_id', axis=1)

netflix.shape[1]
print('There are', netflix[netflix.duplicated(['title', 'country', 'type', 'release_year'])].shape[0], 'rows that are duplicates.')
netflix = netflix.drop_duplicates(['title', 'country', 'type', 'release_year'])

netflix.shape[0]
netflix = netflix.reset_index()

netflix.head()
netflix['rating'] = netflix['rating'].fillna(netflix['rating'].mode()[0])

netflix['rating'].value_counts()
netflix['date_added'] = netflix['date_added'].fillna('January 1, {}'.format(str(netflix['release_year'].mode()[0])))

netflix['date_added'].isnull().sum()
netflix['date_added'] = pd.to_datetime(netflix['date_added'])

netflix.dtypes
# Using a line graph, do we have enough data per year?

netflix.date_added.dt.year.dropna().astype(int).value_counts().sort_index().plot()

plt.show()
# removing rows from 2020 since there is only 18 days of data available

netflix.drop(netflix[netflix.date_added.dt.year == 2020].index, inplace=True)

netflix.shape[0]
# line graph without 2020 information

netflix.date_added.dt.year.dropna().astype(int).value_counts().sort_index().plot()

plt.show()
netflix.type.value_counts()
movie_cnt = len(netflix[netflix.type == 'Movie'])

tv_cnt = len(netflix[netflix.type == 'TV Show'])

print('Percentage of Movies: {:.1f}%'.format((movie_cnt / len(netflix.type)) * 100))

print('Percentage of TV Shows: {:.1f}%'.format((tv_cnt / len(netflix.type)) * 100))
netflix.type.value_counts().plot(kind='pie', autopct='%1.f%%', startangle=90)

plt.show()
# How many titles are added each year?

pd.crosstab(netflix.date_added.dt.year, netflix.type).plot(kind = 'bar')

plt.xlabel('Year Added to Netflix')

plt.ylabel('Number of Titles')

plt.title('Titles Added per Year')

plt.show()
# 'UR' (Unrated) rating is equivalent to 'NR' (Not Rated)

netflix['rating'] = netflix['rating'].replace(to_replace = 'UR', value = 'NR')
netflix['rating'].value_counts().plot(kind='bar')

plt.title('Titles by Rating')

plt.xlabel('Rating')

plt.ylabel('Count')

plt.show()
# Counts by Rating for each Type

plt.figure(figsize = (10, 8))

sns.countplot(x = 'rating', hue = 'type', data = netflix)

plt.title('Counts by Rating for Each Type')

plt.show()
# Counts by Country

netflix['country'].value_counts().sort_values(ascending=False).head()
top_five_countries = netflix[

    (netflix.country == 'United States') | 

    (netflix.country == 'India') | 

    (netflix.country == 'United Kingdom') | 

    (netflix.country == 'Japan') | 

    (netflix.country == 'Canada')]

plt.figure(figsize = (10,8))

sns.countplot(x = 'country', hue = 'type', data = top_five_countries)

plt.title('Count by Top 5 Countries for Each Type')

plt.show()