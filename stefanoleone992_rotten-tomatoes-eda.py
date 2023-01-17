import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)

pd.set_option('display.max_columns', None)

pd.options.mode.chained_assignment = None



movies_df = pd.read_csv('../input/rotten-tomatoes-movies-and-critics-datasets/rotten_tomatoes_movies.csv')

reviews_df = pd.read_csv('../input/rotten-tomatoes-movies-and-critics-datasets/rotten_tomatoes_reviews.csv')



movies_df.head()
reviews_df.head()
movies_df = movies_df[movies_df.in_theaters_date.notnull()]

movies_df['in_theaters_date'] = pd.to_datetime(movies_df['in_theaters_date'])

movies_df['movie_year'] = movies_df['in_theaters_date'].apply(lambda x: x.year)



sns.set(style="white")



plt.figure(figsize=(15,10))

plt.title('Movies by the year', size=20)

sns.distplot(movies_df.movie_year, kde=False)

plt.ylabel('Number of movies', size=15)

plt.xlabel('Year of release',size=15)

plt.axis([1920, 2019, 0, 1750])

plt.xticks(np.arange(1920, 2018, step=5),rotation=45, ha='right')

plt.show()
reviews_df = reviews_df[reviews_df.review_date.notnull()]

reviews_df['review_date'] = pd.to_datetime(reviews_df['review_date'])

reviews_df['review_year'] = reviews_df['review_date'].apply(lambda x: x.year)

reviews_df = reviews_df[reviews_df.review_year.astype(int) >= 2000]



plt.figure(figsize=(15,10))

plt.title('Reviews by the year', size=20)

sns.distplot(reviews_df.review_year, bins=20, kde=False)

plt.ylabel('Number of critic reviews', size=15)

plt.xlabel('Year of review posted',size=15)

plt.axis([2000, 2019, 0, 75000])

plt.xticks(np.arange(2000, 2019, step=1),rotation=45, ha='right')

plt.show()
movies_df = movies_df[(movies_df.tomatometer_rating.notnull()) &

                      (movies_df.audience_rating.notnull())]

sns.jointplot(x=movies_df['movie_year'], y=movies_df['tomatometer_rating'],

              kind="kde").fig.set_size_inches(15,15)
sns.jointplot(x=movies_df['movie_year'], y=movies_df['audience_rating'],

              kind="kde").fig.set_size_inches(15,15)
a = plt.cm.cool



plt.figure(figsize=(15,10))

count = movies_df['studio_name'].value_counts()[:10]

sns.barplot(count.values, count.index, palette=[a(0.1),a(0.2),a(0.3),a(0.4),a(0.5),a(0.6),a(0.7),a(0.8),a(0.9),a(0.99)])

for i, v in enumerate(count.values):

    plt.text(0.8,i,v,color='k',fontsize=14)

plt.xlabel('Count', fontsize=12)

plt.ylabel('Studio name', fontsize=12)

plt.title("Distribution of Studio names", fontsize=16)
movies_df['first_genre'] = movies_df['genre'].str.split(',').str[0]



a = plt.cm.cool



plt.figure(figsize=(15,10))

count = movies_df['first_genre'].value_counts()[:7]

sns.barplot(count.values, count.index, palette=[a(0.1),a(0.2),a(0.3),a(0.4),a(0.5),a(0.6),a(0.7)])

for i, v in enumerate(count.values):

    plt.text(0.8,i,v,color='k',fontsize=14)

plt.xlabel('Count', fontsize=12)

plt.ylabel('Genre name', fontsize=12)

plt.title("Distribution of Genres", fontsize=16)
top_genres = list(count.index)

movie_genres_df = movies_df[movies_df['first_genre'].isin(top_genres)]

movie_genres_df = movie_genres_df[pd.notnull(movie_genres_df[['first_genre', 'tomatometer_rating', 'tomatometer_status', 'tomatometer_count',

                                                              'audience_rating', 'audience_status', 'audience_count']])]



plt.figure(figsize=(15, 10))

sns.boxplot(x='first_genre', y='tomatometer_rating', data=movie_genres_df)

plt.xlabel("Genre Name",fontsize=12)

plt.ylabel("TomatoMeter Rating",fontsize=12)

plt.title("Boxplot of TomatoMeter rating per Genre", fontsize=16)

plt.show()
plt.figure(figsize=(15, 10))

sns.boxplot(x='first_genre', y='audience_rating', data=movie_genres_df)

plt.xlabel("Genre Name",fontsize=12)

plt.ylabel("Audience Rating",fontsize=12)

plt.title("Boxplot of Audience rating per Genre", fontsize=16)

plt.show()
genre_rating_tomatometer = pd.crosstab(movie_genres_df.first_genre, movie_genres_df.tomatometer_status, margins=True)

genre_rating_tomatometer.style.background_gradient(cmap='summer_r')
genre_rating_audience = pd.crosstab(movie_genres_df.first_genre, movie_genres_df.audience_status, margins=True)

genre_rating_audience.style.background_gradient(cmap='summer_r')
sns.factorplot('first_genre', 'tomatometer_count', hue='tomatometer_status', data=movie_genres_df)

fig = plt.gcf()

fig.set_size_inches(20, 8)

plt.xlabel("Genre Name",fontsize=12)

plt.ylabel("TomatoMeter Count",fontsize=12)

plt.title("Factorplots of Genres and TomatoMeter data", fontsize=16)

plt.show()
sns.factorplot('first_genre', 'audience_count', hue='audience_status', data=movie_genres_df)

fig = plt.gcf()

fig.set_size_inches(20, 8)

plt.xlabel("Genre Name",fontsize=12)

plt.ylabel("Audience Count",fontsize=12)

plt.title("Factorplots of Genres and Audience data", fontsize=16)

plt.show()
f,ax = plt.subplots(3,1, figsize=(15, 30))

sns.distplot(movie_genres_df[(movie_genres_df['tomatometer_status'] == 'Certified Fresh') &

                             (movie_genres_df['tomatometer_count'] <= 400)].tomatometer_count, ax=ax[0], bins=30)

ax[0].set_title('TomatoMeter count in Certified Fresh', fontsize=16)

ax[0].set_xlabel("TomatoMeter Count",fontsize=12)

ax[0].set_xlim([0,400])

sns.distplot(movie_genres_df[(movie_genres_df['tomatometer_status'] == 'Fresh') &

                             (movie_genres_df['tomatometer_count'] <= 400)].tomatometer_count, ax=ax[1], bins=30)

ax[1].set_title('TomatoMeter count in Fresh', fontsize=16)

ax[1].set_xlabel("TomatoMeter Count",fontsize=12)

ax[1].set_xlim([0,400])

sns.distplot(movie_genres_df[(movie_genres_df['tomatometer_status'] == 'Rotten') &

                             (movie_genres_df['tomatometer_count'] <= 400)].tomatometer_count, ax=ax[2], bins=30)

ax[2].set_title('TomatoMeter count in Rotten', fontsize=16)

ax[2].set_xlabel("TomatoMeter Count",fontsize=12)

ax[2].set_xlim([0,400])

plt.show()
f,ax = plt.subplots(2,1, figsize=(15, 20))

sns.distplot(movie_genres_df[(movie_genres_df['audience_status'] == 'Upright') &

                             (movie_genres_df['audience_count'] <= 10000)].audience_count, ax=ax[0], bins=30)

ax[0].set_title('Audience count in Upright', fontsize=16)

ax[0].set_xlabel("Audience Count",fontsize=12)

ax[0].set_xlim([0,10000])

sns.distplot(movie_genres_df[(movie_genres_df['audience_status'] == 'Spilled') &

                             (movie_genres_df['audience_count'] <= 10000)].audience_count, ax=ax[1], bins=30)

ax[1].set_title('Audience count in Spilled', fontsize=16)

ax[1].set_xlabel("Audience Count",fontsize=12)

ax[1].set_xlim([0,10000])

plt.show()
group_names = movie_genres_df.first_genre.value_counts().head(7).index

group_size = movie_genres_df.first_genre.value_counts().head(7)

subgroup_names = ['CertFresh','Fresh','Rotten', 'CertFresh','Fresh','Rotten', 'CertFresh','Fresh','Rotten', 'CertFresh', 'Fresh', 'Rotten',

                  'CertFresh','Fresh','Rotten', 'CertFresh','Fresh','Rotten', 'CertFresh','Fresh','Rotten']

size_list = []

for element in group_names:

    size_list.append(genre_rating_tomatometer.loc[element]['Certified Fresh'])

    size_list.append(genre_rating_tomatometer.loc[element]['Fresh'])

    size_list.append(genre_rating_tomatometer.loc[element]['Rotten'])

subgroup_size = size_list



fig, ax = plt.subplots()

ax.axis('equal')

outter_pie, _ = ax.pie(group_size, radius=4, labels=group_names,

                       colors=[a(0.1),a(0.2),a(0.3),a(0.4),a(0.5),a(0.6),a(0.7)])

plt.setp(outter_pie, width=1, edgecolor='white') 

inner_pie, _ = ax.pie(subgroup_size, radius=3, labels=subgroup_names, labeldistance=0.83,

                      colors=['green','gold','red', 'green','gold','red', 'green','gold','red', 'green','gold','red',

                              'green','gold','red', 'green','gold','red', 'green','gold','red'])

plt.setp(inner_pie, width=0.4, edgecolor='white')

plt.margins(0,0)

plt.show()
group_names = movie_genres_df.first_genre.value_counts().head(7).index

group_size = movie_genres_df.first_genre.value_counts().head(7)

subgroup_names = ['Upright','Spilled', 'Upright','Spilled', 'Upright','Spilled', 'Upright','Spilled',

                  'Upright','Spilled', 'Upright','Spilled', 'Upright','Spilled']

size_list = []

for element in group_names:

    size_list.append(genre_rating_audience.loc[element]['Upright'])

    size_list.append(genre_rating_audience.loc[element]['Spilled'])

subgroup_size = size_list



fig, ax = plt.subplots()

ax.axis('equal')

outter_pie, _ = ax.pie(group_size, radius=4, labels=group_names,

                       colors=[a(0.1),a(0.2),a(0.3),a(0.4),a(0.5),a(0.6),a(0.7)])

plt.setp(outter_pie, width=1, edgecolor='white') 

inner_pie, _ = ax.pie(subgroup_size, radius=3, labels=subgroup_names, labeldistance=0.83,

                      colors=['green','red', 'green','red', 'green','red', 'green','red',

                              'green','red', 'green','red', 'green','red'])

plt.setp(inner_pie, width=0.4, edgecolor='white')

plt.margins(0,0)

plt.show()