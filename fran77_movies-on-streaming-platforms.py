# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

from collections import Counter

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv')
data = data.iloc[:,1:]
data.head()
total_platforms = data[['Netflix', 'Hulu', 'Prime Video', 'Disney+']].apply(pd.Series.value_counts).reset_index()

total_platforms.columns = ['Present', 'Netflix', 'Hulu', 'Prime Video', 'Disney+']
total_platforms
labels = ['Netflix', 'Hulu', 'Prime Video', 'Disney+']
plt.figure(figsize=(12,8))

p = plt.pie(total_platforms[total_platforms.Present == 1].iloc[:,1:], labels=labels, explode = (0.1, 0.1, 0.1, 0.1), 

            autopct='%1.0f%%', labeldistance=1.1, shadow=True)
for i in labels:

    data['rating_'+i] = data['IMDb'] * data[i]
imdb_platforms = data.replace(0, np.nan)
imdb_platforms = imdb_platforms[['rating_Netflix', 'rating_Hulu', 'rating_Prime Video', 'rating_Disney+']].mean().reset_index()

imdb_platforms.columns = ['ratings', 'mean']

imdb_platforms.sort_values(by='mean', ascending=False)
plt.figure(figsize=(12,8))

ax = sns.barplot(x="ratings", y="mean", data=imdb_platforms)

l = ax.set(ylim=(5.5, 6.5))
# Prime Video has more films but less quality types
fig, axes = plt.subplots(2, 2,figsize=(22,15))



ax1 = data[data.Netflix == 1].groupby('IMDb')['ID'].count().plot(xlim=(0, 10), ax=axes[0,0])

ax2 = data[data.Hulu == 1].groupby('IMDb')['ID'].count().plot(xlim=(0, 10),ax=axes[0,1])

ax3 = data[data['Prime Video'] == 1].groupby('IMDb')['ID'].count().plot(xlim=(0, 10),ax=axes[1,0])

ax4 = data[data['Disney+'] == 1].groupby('IMDb')['ID'].count().plot(xlim=(0, 10),ax=axes[1,1])



ax1.title.set_text(labels[0])

ax2.title.set_text(labels[1])

ax3.title.set_text(labels[2])

ax4.title.set_text(labels[3])



plt.show()
# Prime Video has a larger proportion of movies rated under 4
fig, axes = plt.subplots(2, 2,figsize=(22,15))



ax1 = data[data.Netflix == 1].groupby('Year')['ID'].count().plot(xlim=(data['Year'].min(), data['Year'].max()), ax=axes[0,0])

ax2 = data[data.Hulu == 1].groupby('Year')['ID'].count().plot(xlim=(data['Year'].min(), data['Year'].max()),ax=axes[0,1])

ax3 = data[data['Prime Video'] == 1].groupby('Year')['ID'].count().plot(xlim=(data['Year'].min(), data['Year'].max()),ax=axes[1,0])

ax4 = data[data['Disney+'] == 1].groupby('Year')['ID'].count().plot(xlim=(data['Year'].min(), data['Year'].max()),ax=axes[1,1])



ax1.title.set_text(labels[0])

ax2.title.set_text(labels[1])

ax3.title.set_text(labels[2])

ax4.title.set_text(labels[3])



plt.show()
# Netflix, Hulu and Prime Video (which has some bumps in the 40s and 80s) have their peak in the 2010s

# Disney+ in the 2000s and in the late 2010s
fig = plt.figure(figsize=(35,15))



age_netflix = data[data.Netflix == 1].groupby(['Age', 'Netflix']).count()['ID'].reset_index()[['Age', 'ID']]

age_hulu = data[data.Hulu == 1].groupby(['Age', 'Hulu']).count()['ID'].reset_index()[['Age', 'ID']]

age_prime = data[data['Prime Video'] == 1].groupby(['Age', 'Prime Video']).count()['ID'].reset_index()[['Age', 'ID']]

age_disney = data[data['Disney+'] == 1].groupby(['Age', 'Disney+']).count()['ID'].reset_index()[['Age', 'ID']]



ax = plt.subplot2grid((2,4),(0,0))

p = plt.pie(age_netflix['ID'], labels=age_netflix.Age, explode = (0.05, 0.05, 0.05, 0.05, 0.05), autopct='%1.0f%%', labeldistance=1.1, shadow=True)

plt.title('Netflix')

ax = plt.subplot2grid((2,4),(0,1))

p = plt.pie(age_hulu['ID'], labels=age_hulu.Age, explode = (0.05, 0.05, 0.05, 0.05, 0.05), autopct='%1.0f%%', labeldistance=1.1, shadow=True)

plt.title('Hulu')

ax = plt.subplot2grid((2,4),(1,0))

p = plt.pie(age_prime['ID'], labels=age_prime.Age, explode = (0.05, 0.05, 0.05, 0.05, 0.05), autopct='%1.0f%%', labeldistance=1.1, shadow=True)

plt.title('Prime Video')

ax = plt.subplot2grid((2,4),(1,1))

p = plt.pie(age_disney['ID'], labels=age_disney.Age, explode = (0.05, 0.05, 0.05, 0.05, 0.05), autopct='%1.0f%%', labeldistance=1.1, shadow=True)

plt.title('Disney+')

plt.show()
# Disney has a lot more accessible contents (91% of all or 7+)
# Getting the same colours

nb_genres = data[['ID', 'Genres']]

nb_genres = nb_genres.dropna()

nb_genres['Genres'] = nb_genres['Genres'].astype(str)

nb_genres = pd.concat([pd.Series(row['ID'], row['Genres'].split(','))              

                    for _, row in nb_genres.iterrows()]).reset_index()

nb_genres = nb_genres.groupby('index').count().reset_index()

nb_genres.columns = ['Genres', 'nb']

nb_genres = nb_genres.sort_values(by='nb', ascending=False)

nb_genres['colors'] = sns.color_palette(n_colors=len(nb_genres))
genres_plat = data[['Netflix', 'Hulu', 'Prime Video', 'Disney+', 'Genres']]

genres_plat = genres_plat.dropna()

genres_plat['Genres'] = genres_plat['Genres'].astype(str)



# Netflix

genres_netflix = genres_plat[genres_plat.Netflix == 1]

genres_netflix = pd.concat([pd.Series(row['Netflix'], row['Genres'].split(','))              

                    for _, row in genres_netflix.iterrows()]).reset_index()

genres_netflix.columns = ['Genres', 'Count']

genres_netflix = genres_netflix.groupby('Genres').count().reset_index().sort_values(by='Count', ascending=False)

genres_netflix['Count'] = genres_netflix['Count'] / len(genres_netflix)



# Hulu

genres_hulu = genres_plat[genres_plat.Hulu == 1]

genres_hulu = pd.concat([pd.Series(row['Hulu'], row['Genres'].split(','))              

                    for _, row in genres_hulu.iterrows()]).reset_index()

genres_hulu.columns = ['Genres', 'Count']

genres_hulu = genres_hulu.groupby('Genres').count().reset_index().sort_values(by='Count', ascending=False)

genres_hulu['Count'] = genres_hulu['Count'] / len(genres_hulu)



# Prime Video

genres_prime = genres_plat[genres_plat['Prime Video'] == 1]

genres_prime = pd.concat([pd.Series(row['Prime Video'], row['Genres'].split(','))              

                    for _, row in genres_prime.iterrows()]).reset_index()

genres_prime.columns = ['Genres', 'Count']

genres_prime = genres_prime.groupby('Genres').count().reset_index().sort_values(by='Count', ascending=False)

genres_prime['Count'] = genres_prime['Count'] / len(genres_prime)



# Disney+

genres_disney = genres_plat[genres_plat['Disney+'] == 1]

genres_disney = pd.concat([pd.Series(row['Disney+'], row['Genres'].split(','))              

                    for _, row in genres_disney.iterrows()]).reset_index()

genres_disney.columns = ['Genres', 'Count']

genres_disney = genres_disney.groupby('Genres').count().reset_index().sort_values(by='Count', ascending=False)

genres_disney['Count'] = genres_disney['Count'] / len(genres_disney)
genres_netflix = pd.merge(genres_netflix, nb_genres, on='Genres')

genres_hulu = pd.merge(genres_hulu, nb_genres, on='Genres')

genres_prime = pd.merge(genres_prime, nb_genres, on='Genres')

genres_disney = pd.merge(genres_disney, nb_genres, on='Genres')
fig = plt.figure(figsize=(35,15))

ax = plt.subplot2grid((2,4),(0,0))

p = plt.pie(genres_netflix['Count'][:10], labels=genres_netflix.Genres[:10], colors = genres_netflix.colors[:10], explode = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05), autopct='%1.0f%%', labeldistance=1.1, shadow=True)

plt.title('Netflix')

ax = plt.subplot2grid((2,4),(0,1))

p = plt.pie(genres_hulu['Count'][:10], labels=genres_hulu.Genres[:10], colors = genres_hulu.colors[:10], explode = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05), autopct='%1.0f%%', labeldistance=1.1, shadow=True)

plt.title('Hulu')

ax = plt.subplot2grid((2,4),(1,0))

p = plt.pie(genres_prime['Count'][:10], labels=genres_prime.Genres[:10], colors = genres_prime.colors[:10], explode = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05), autopct='%1.0f%%', labeldistance=1.1, shadow=True)

plt.title('Prime Video')

ax = plt.subplot2grid((2,4),(1,1))

p = plt.pie(genres_disney['Count'][:10], labels=genres_disney.Genres[:10], colors = genres_disney.colors[:10], explode = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05), autopct='%1.0f%%', labeldistance=1.1, shadow=True)

plt.title('Disney+')

plt.show()

# Netflix, Hulu and Prime Video have nearly the same top genres

# More movies with Adventure and Family in Disney+, less drama than the others
plt.figure(figsize=(20,8))

g = sns.countplot(data.Year)

t = plt.xticks(rotation=90)
# Peak in 2017 because platforms have to wait 2 years after they've been in theaters before having them in streaming
# Better way to see it as a percentage for the years
genres = data[['Year', 'Genres']]

genres = genres.dropna()

genres['Genres'] = genres['Genres'].astype(str)
genres = pd.concat([pd.Series(row['Year'], row['Genres'].split(','))              

                    for _, row in genres.iterrows()]).reset_index()

genres.columns = ['Genres', 'Year']
total_year = genres.groupby('Year').count()

total_year = total_year.reset_index()

total_year.columns = ['Year', 'total']
genres_year = genres.groupby(['Year', 'Genres'])['Genres'].count()

genres_year = genres_year.reset_index(name="count")
genres_year = genres_year.loc[genres_year.groupby('Year')['count'].nlargest(1).index.get_level_values(1)]
genres_year = pd.merge(genres_year, total_year, on='Year')

genres_year['perc_mc'] = genres_year['count'] / genres_year['total']
fig,ax = plt.subplots()

fig.set_size_inches(20,10)

ax = sns.barplot(x='Year',y='perc_mc',hue='Genres', data=genres_year, dodge=False, palette='Paired')

plt.xticks(rotation=90)

ax.legend(loc='upper right')

plt.show()
# Seems like a lot of Drama movies
plt.figure(figsize=(20,8))

g = sns.countplot(data.IMDb)

t = plt.xticks(rotation=90)
# What are the best rated movies in all platform ?

data[data.IMDb == data.IMDb.max()][['Title', 'IMDb', 'Genres', 'Runtime']]
# A lot of Documentaries

# This ratings don't seem accurate : My Next Guest... has a 7.9 rating and Bounty 7 after checking

# The rest is correct or has less than 0.3 difference
age = data.groupby('Age').count()['ID'].reset_index()

age.columns = ['Age', 'count']
age.sort_values(by='count', ascending=False)
plt.figure(figsize=(12,8))

p = plt.pie(age['count'], labels=age.Age, explode = (0.05, 0.05, 0.05, 0.05, 0.05), autopct='%1.0f%%', labeldistance=1.1, shadow=True)
# before 1960 / 1960-80 / 1980-2000 / 2000+
ratings = data[['Year', 'IMDb']]

ratings = ratings.dropna()
for i, row in ratings.iterrows():

    if row['Year'] < 1960:

        ratings.loc[i,'period'] = '-1960'

    elif ((row['Year'] >=1960) &  (row['Year'] <1980)):

        ratings.loc[i,'period'] = '1960-80'

    elif ((row['Year'] >=1980) &  (row['Year'] <2000)):

        ratings.loc[i,'period'] = '1980-2000'

    else:

        ratings.loc[i,'period'] = '2000+'
ratings = ratings.groupby('period')['IMDb'].mean().reset_index()

ratings.columns = ['period', 'mean']
plt.figure(figsize=(12,8))

ax = sns.barplot(x="period", y="mean", data=ratings)

l = ax.set(ylim=(5.5, 6.2))
countries = data[['Country']]

countries = countries.dropna()

countries = pd.concat([pd.Series(row['Country'].split(','))              

                    for _, row in countries.iterrows()]).reset_index()

countries.columns = ['index', 'Country']
plt.figure(figsize=(15,8))

g = sns.countplot(x='Country',data=countries, order = countries['Country'].value_counts().iloc[:10].index)
# Majority of movies are made in the US

# So we suppose majority of the movies are in English
language = data[['Language']]

language = language.dropna()

language = pd.concat([pd.Series(row['Language'].split(','))              

                    for _, row in language.iterrows()]).reset_index()

language.columns = ['index', 'Language']
plt.figure(figsize=(15,8))

g = sns.countplot(x='Language',data=language, order = language['Language'].value_counts().iloc[:10].index)
data['Runtime'].mean()
plt.figure(figsize=(10,6))

sns.kdeplot(data['Runtime'])
# Big Right Skew

# How many movies with more than 200 min  (3h+)?
print('%.2f%% movies have more than 300 min runtime' %(data[data.Runtime > 200].count()['ID'] / len(data) *100))
# Colorado with 20h+

data[data.Runtime == data['Runtime'].max()][['Title', 'Runtime']]
# Let's remove this noise

runtime = data[data.Runtime < 200]
plt.figure(figsize=(15,8))

sns.distplot(runtime['Runtime'], bins=50, kde=False)
print('The most frequent runtime is %d min' % data['Runtime'].mode().values[0])