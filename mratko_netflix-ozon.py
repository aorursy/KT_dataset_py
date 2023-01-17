# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')

df.head()
df.info()
df.show_id.unique().shape
print(df.show_id.min(), df.show_id.max(), df.show_id.max() - df.show_id.min())
plt.hist(df.show_id, bins=10)

plt.title('show_id histogram bins = 10')
plt.hist(df.show_id, bins=20)

plt.title('show_id histogram bins = 20')
types = df.type.value_counts()

plt.bar(types.index, types)

plt.title('number of movies and tv shows')
print(df.title.size, df.title.unique().size, df.title.size - df.title.unique().size)
df_title_doublicates = df[df.title.duplicated(keep=False) == True].sort_values(by='title')

df_title_doublicates
df.director.unique().size, df.director.size
df.director.value_counts()[:20]
df.country.unique().size
df.country.value_counts()
df.country.value_counts()[:50]
df_top12_countries = df.country.value_counts()[:12]

df_top12_countries
df_top12_countries.plot(kind='bar', figsize=(10,5))

plt.ylabel('number of movies/shows')

plt.title('top 12 countries')
df_date_added = df.date_added.value_counts()

df_date_added[:10]
df['datetime_added'] = pd.to_datetime(df.date_added)

print(df['datetime_added'].min(), df['datetime_added'].max())

df['datetime_added'].hist(bins = 20)

plt.title('histogram number of movies/films by date_added bins = 20')
df['datetime_added'].hist(bins = 200, figsize=(10, 5))

plt.title('histogram number of movies/films by date_added bins = 200')
df['month_added'] = df['datetime_added'].dt.month

df['year_added'] = df['datetime_added'].dt.year

df['day_added'] = df['datetime_added'].dt.day
df.month_added.hist(bins=12)

plt.title('histogram number of movies/shows by months')
df.day_added.hist(bins=30)

plt.title('histogram number of movies/shows by days bins=30')
df.year_added.hist(bins=13)

plt.title('histogram number of movies/shows by year')
df.year_added[df.year_added < 2016].hist(bins=8)

plt.title('histogram number of movies/shows in 2008-2015')
df.release_year.hist()

plt.title('histogram number of movies/shows by release year')
df.release_year[df.release_year > 2006].hist(bins=14)

plt.title('histogram number of movies/shows by release year')
(df.year_added - df.release_year).value_counts()
(df.year_added - df.release_year).hist(bins=50, figsize=(8, 6))

plt.xlabel('difference between year_added and release_year')

plt.title('Гистограмма')
df[(df.year_added - df.release_year) < 0]
df[(df.year_added - df.release_year) < 0].shape
ratings = df.rating.value_counts()

ratings.plot(kind='bar', figsize=(10,5))

#plt.bar(ratings.index, ratings, figsize=(10, 5))

plt.title('number of movies and tv shows by rating')
sorted_rating = ['G', 'TV-Y', 'TV-G', 'PG', 'TV-Y7', 'TV-Y7-FV',

                 'TV-PG', 'PG-13', 'TV-14', 'R', 'NC-17', 'TV-MA']

colors = ['y', 'y', 'y', 'g', 'g', 'g', 'g', 'b', 'b', 'r', 'r', 'r']

ratings = df.rating.value_counts()[sorted_rating]

f, ax = plt.subplots(figsize=(10,5))

plt.bar(ratings.index, ratings, color=colors)



import matplotlib.patches as mpatches



red_patch = mpatches.Patch(color='r', label='Mature')

blue_patch = mpatches.Patch(color='b', label='Teens')

green_patch = mpatches.Patch(color='g', label='Older Kids')

yellow_patch = mpatches.Patch(color='y', label='Little Kids')

plt.legend(handles=[yellow_patch, green_patch, blue_patch, red_patch], prop={'size': 12})

plt.title('number of movies and tv shows by rating')
duration_movies = df[df.type == 'Movie'].duration.str[:-3].astype(float)

duration_movies.hist(bins=50)

plt.title('duration of movies in minutes')
duration_movies.describe()
duration_series = df[df.type == 'TV Show'].duration.str[:-7].astype(float)

duration_series.describe()
duration_series.hist(bins=15)

plt.title('number of seasons of TV Shows')
genres_top10 = df.listed_in.value_counts()[:10]

plt.barh(genres_top10.index, genres_top10)

plt.title('number of movies and tv shows by genres')
df[df.listed_in.str.find(',') < 0].listed_in.unique()
unique_genres = ['Stand-Up Comedy', "Kids' TV", 'Comedies',

       'Children & Family Movies', 'Documentaries', 'Docuseries',

       'Horror Movies', 'TV Comedies', 'Action & Adventure', 'Movies',

       'Reality TV', 'Dramas', 'Thrillers', 'International Movies',

       'International TV Shows', 'Music & Musicals', 'Anime Features',

       'Anime Series', 'TV Shows', 'TV Dramas', 'Sports Movies',

       'Romantic Movies', 'Stand-Up Comedy & Talk Shows',

       'Independent Movies', 'Sci-Fi & Fantasy', 'TV Action & Adventure']



count_genre = []

for genre in unique_genres:

    count_genre.append(sum(df.listed_in.str.count(genre)))

    

count_genre_df  = pd.DataFrame({'genre': unique_genres, 'quantity': count_genre})

count_genre_df.sort_values(by=['quantity']).plot.barh(x='genre', y='quantity', figsize=(10,10))

plt.title('Popularity of genres')
unique_genres = ['Stand-Up Comedy', "Kids' TV", 'Comedies',

       'Children & Family Movies', 'Documentaries', 'Docuseries',

       'Horror Movies', 'TV Comedies', 'Action & Adventure', 

       'Reality TV', 'Dramas', 'Thrillers', 'International Movies',

       'International TV Shows', 'Music & Musicals', 'Anime Features',

       'Anime Series', 'TV Dramas', 'Sports Movies',

       'Romantic Movies', 'Stand-Up Comedy & Talk Shows',

       'Independent Movies', 'Sci-Fi & Fantasy', 'TV Action & Adventure']



count_genre = []

for genre in unique_genres:

    count_genre.append(sum(df.listed_in.str.count(genre)))

    

count_genre_df  = pd.DataFrame({'genre': unique_genres, 'quantity': count_genre})

count_genre_df.sort_values(by=['quantity']).plot.barh(x='genre', y='quantity', figsize=(10,10))

plt.title('Popularity of genres')
plt.scatter(df.show_id, df.type)

plt.title('scatter plot for show_id and type of show')

plt.xlabel('show_id')
plt.scatter(df[df.type == 'Movie'].show_id, df[df.type == 'Movie'].release_year)

plt.title('scatter plot for show_id and release_year of movies')

plt.xlabel('show_id')
plt.scatter(df[df.type == 'TV Show'].show_id, df[df.type == 'TV Show'].release_year)

plt.title('scatter plot for show_id and release_year of TV shows')

plt.xlabel('show_id')
plt.scatter(df[df.type == 'Movie'].show_id, df[df.type == 'Movie'].year_added)

plt.title('scatter plot for show_id and year_added of movies')

plt.xlabel('show_id')
plt.scatter(df[df.type == 'TV Show'].show_id, df[df.type == 'TV Show'].year_added)

plt.title('scatter plot for show_id and year_added of TV Show')

plt.xlabel('show_id')
df[(df.type == 'TV Show') & (df.show_id < 70000000)]
df[(df.type == 'TV Show') & (df.year_added == 2008)]
'''

import seaborn as sns

df1 = df[df.type == 'TV Show']

sns.stripplot(x='year_added', y='show_id', data=df[df.type == 'TV Show'], jitter=True)



sns.despine()

'''
pd.crosstab(df.type, df.year_added)
df1 = df[(df.year_added > 2013) & (df.year_added < 2020)]
sum(df1[df1.type == 'Movie'].listed_in.str.count('TV Comedies'))
sum(df1[df1.type == 'Movie'].listed_in.str.count('TV Dramas'))
sum(df1[df1.type == 'TV Show'].listed_in.str.count('Comedies'))
genres_top5 = ['Comedies', 'Documentaries', 'Action & Adventure', 'Dramas', 'Thrillers']

years = [2014, 2015, 2016, 2017, 2018, 2019]

res = pd.DataFrame()

#res = pd.DataFrame(columns=('genre', 'year'))

for genre in genres_top5:

        for year in years:

            res.loc[genre, year] =  sum(df1[df1.year_added == year].listed_in.str.count(genre))

res
ind = range(6) 

comedy = [5, 23, 98, 278, 431, 634]

doc = [8, 13, 70, 210, 172, 191]

act = [1, 4, 28, 133, 216, 302]

drama = [3, 21, 117, 474, 681, 850]

thriller = [1, 3, 21, 81, 129, 178]

width = 0.35

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.bar(ind, drama, width, color='r')

ax.bar(ind, comedy, width,bottom=drama, color='b')

ax.bar(ind, act, width, bottom=[sum(x) for x in zip(drama, comedy)], color='g')

ax.bar(ind, doc, width, bottom=[sum(x) for x in zip(drama, comedy, act)], color='y')

ax.bar(ind, thriller, width, bottom=[sum(x) for x in zip(drama, comedy, act, doc)], color='black')

ax.set_ylabel('Number of Movies')

ax.set_title('Most popular genres by year')

ax.set_xticks(ind, ('2014', '2015', '2016', '2017', '2018', '2019'))

#ax.set_yticks(np.arange(0, 81, 10))

ax.legend(labels=['Drama', 'Comedy', 'Action&Adventure', 'Documentaries', 'Thirller'])

plt.show()
show_month = df1[df1.type == 'TV Show'].groupby('month_added').count()

movie_month = df1[df1.type == 'Movie'].groupby('month_added').count()

#ax = plt.subplot(111)

fig, ax = plt.subplots(figsize=(10, 6))

months = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

months_shift = [sum(x) for x in zip(months, [- 0.4] * 12)] 



ax.bar(months_shift, show_month['show_id'], width=0.4, color='b', align='center')

ax.bar(months, movie_month['show_id'], width=0.4, color='g', align='center')

ax.set_ylabel('Number of Movies/TV Shows')

ax.legend(labels=['TV Show', 'Movie'])

ax.set_xticks(months)

ax.set_xticklabels(['January', 'February', 'March', 'April', 'May',

                          'June', 'July', 'August', 'September', 'October', 'November', 'December'], rotation=60)

plt.title('Number of TV Shows/Movies by months')

plt.show()
show_years= df1[df1.type == 'TV Show'].groupby('year_added').count()

movie_years = df1[df1.type == 'Movie'].groupby('year_added').count()

#ax = plt.subplot(111)

fig, ax = plt.subplots(figsize=(10, 6))

years = [2014, 2015, 2016, 2017, 2018, 2019]

years_shift = [sum(x) for x in zip(years, [- 0.4] * len(years))] 



ax.bar(years_shift, show_years['show_id'], width=0.4, color='b')

ax.bar(years, movie_years['show_id'], width=0.4, color='g')

ax.set_ylabel('Number of Movies/TV Shows')

ax.legend(labels=['TV Show', 'Movie'])



plt.title('Number of TV Shows/Movies by years')

plt.show()
little_kids = ['G', 'TV-Y', 'TV-G']

older_kids = [ 'PG', 'TV-Y7', 'TV-Y7-FV', 'TV-PG']

teens = ['PG-13', 'TV-14']

mature = ['R', 'NC-17', 'TV-MA']

df_little = df1[(df1.rating == 'G') | (df1.rating == 'TV-Y') | (df1.rating == 'TV-G')]

df_older = df1[(df1.rating == 'PG') | (df1.rating == 'TV-Y7') | (df1.rating == 'TV-PG')| (df1.rating == 'TV-Y7-FV')]

df_teens = df1[(df1.rating == 'PG-13') | (df1.rating == 'TV-14') ]

df_mature = df1[(df1.rating == 'R') | (df1.rating == 'NC-17') | (df1.rating == 'TV-MA')]
fig, ax = plt.subplots(figsize=(10, 6))

years = [2014, 2015, 2016, 2017, 2018, 2019]

years_shift1 = [sum(x) for x in zip(years, [- 0.2] * len(years))] 

years_shift2 = [sum(x) for x in zip(years_shift1, [- 0.2] * len(years))] 

years_shift3 = [sum(x) for x in zip(years, [0.2] * len(years))] 



ax.bar(years_shift2, df_little.groupby('year_added').count()['show_id'], width=0.2, color='y')

ax.bar(years_shift1, df_older.groupby('year_added').count()['show_id'], width=0.2, color='g')

ax.bar(years, df_teens.groupby('year_added').count()['show_id'], width=0.2, color='b')

ax.bar(years_shift3, df_mature.groupby('year_added').count()['show_id'], width=0.2, color='r')



ax.legend(labels = ['Little kids', 'Older kids', 'Teens', 'Mature'])



ax.set_ylabel('Number of Movies & TV Shows')

plt.title('Number of TV Shows & Movies by years and ratings')

plt.show()
little_tv = df_little[df_little.type == 'TV Show'].count().show_id

little_mov = df_little[df_little.type == 'Movie'].count().show_id



older_tv = df_older[df_older.type == 'TV Show'].count().show_id

older_mov = df_older[df_older.type == 'Movie'].count().show_id



teens_tv = df_teens[df_teens.type == 'TV Show'].count().show_id

teens_mov = df_teens[df_teens.type == 'Movie'].count().show_id



mature_tv = df_mature[df_mature.type == 'TV Show'].count().show_id

mature_mov = df_mature[df_mature.type == 'Movie'].count().show_id



#sum_tv = little_tv + older_tv + teens_tv + mature_tv

#sum_mov = little_mov + older_mov + teens_mov + mature_mov





rate = [0, 1, 2, 3]

rate_shift = [sum(x) for x in zip(rate, [- 0.4] * len(rate))] 



fig, ax = plt.subplots(figsize=(9, 6))

ax.bar(rate, [little_tv/(little_tv + little_mov) * 100, older_tv/(older_tv + older_mov) * 100,

              teens_tv/(teens_tv + teens_mov) * 100, mature_tv/(mature_tv + mature_mov) * 100] , width=0.4, color='b')

ax.bar(rate_shift, [little_mov/(little_tv + little_mov) * 100, older_mov/(older_tv + older_mov)  * 100,

                    teens_mov/(teens_tv + teens_mov) * 100, mature_mov/(mature_tv + mature_mov) * 100], width=0.4, color='g')



ax.legend(labels = ['TV Shows', 'Movies'])



ax.set_xticks(rate)

ax.set_xticklabels(['Little kids', 'Older kids', 'Teens', 'Mature'])

ax.set_ylabel('Percent of content')

plt.title('Percent TV Shows and Movies by all content by different ratings')

plt.show()