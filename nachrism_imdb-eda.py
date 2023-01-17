#Import the tools for analysis

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd

import seaborn as sns

sns.set(style='white', color_codes=True, font_scale=1.25)

import itertools
#Import the data into a DF and view the first 10 entries

imdb = pd.read_csv('../input/IMDB-Movie-Data.csv')

imdb.head()
#Determine the amount of movies that were produced each year, from 2006-2016

movie_yearly_count = imdb['Year'].value_counts().sort_index().plot(kind='bar', color='r', alpha=0.5, grid=False, rot=45)

movie_yearly_count.set_xlabel('Year')

movie_yearly_count.set_ylabel('Movie Count')

movie_yearly_count.set_title('Movie Count by Year')
#Create an array that contains the column names for what we want to compare the ratings to

#in order to see if ratings have any correlation with other aspects of the movies

ratings_comparisons = ['Revenue (Millions)', 'Metascore', 'Runtime (Minutes)', 'Votes']
#Created a for loop that iterates through each of the comparisons made above and plots them separately

for comparison in ratings_comparisons:

    sns.jointplot(x='Rating', y=comparison, data=imdb, alpha=0.5, color='r', size=10, space=0)
#Since most of these films in the data set are part of multiple genres, we need to get a list

#of entirely unique genres, without repeats, to see how many genres we are truly dealing with



unique_genres = imdb['Genre'].unique()

individual_genres = []

for genre in unique_genres:

    individual_genres.append(genre.split(','))



individual_genres = list(itertools.chain.from_iterable(individual_genres))

individual_genres = set(individual_genres)



individual_genres
#Now we can iterate through each genre, counting the number of films that contain that genre

#then plot how many films of each genre were made by year onto a bar graph



print('Number of movies in each genre: \n')



for genre in individual_genres:

    current_genre = imdb['Genre'].str.contains(genre).fillna(False)

    plt.figure()

    plt.xlabel('Year')

    plt.ylabel('Number of Movies Made')

    plt.title(str(genre))

    imdb[current_genre].Year.value_counts().sort_index().plot(kind='bar', color='r', alpha=0.5, rot=45)

    print(genre, len(imdb[current_genre]))
#Determine the percent of total entries are attributed to each genre. Keep in mind that 

#since some films have multiple genres, these percentages won't add to 100%



genre_pcts = np.zeros(len(individual_genres))

i = 0

for genre in individual_genres:

    current_genre = imdb['Genre'].str.contains(genre).fillna(False)

    pct = len(imdb[current_genre]) / 1000 * 100

    genre_pcts[i] = pct

    i += 1

    print(genre, pct)
#Throw our genre percentage values into a DF for easy plotting



genre_pcts_df = pd.DataFrame(genre_pcts, index=individual_genres, columns=['Percent'])

genre_pcts_df
#Taking a sum of the total movies made from the top 5 genres,

#which contributed to the most in terms of movies made?



explode = (0.05, 0.05, 0.08, 0.1, 0.12)

colors = ['#ff3232', '#ff4c4c', '#ff6666', '#ff7f7f', '#ff9999', ]

genre_pcts_df.sort_values(by='Percent', ascending=False).head(5).plot.pie(legend=False, subplots=True, autopct='%.2f%%', figsize=(8,8), colors=colors, explode=explode)

plt.ylabel('')

plt.title('Percent of Total Movies Made from Top 5 Genres', weight='bold', fontsize=16)
#Same idea as above, but we can determine revenue percentage



genre_revenue_pcts = np.zeros(len(individual_genres))

i = 0

for genre in individual_genres:

    current_genre = imdb['Genre'].str.contains(genre).fillna(False)

    revenue_pct = imdb[current_genre].xs('Revenue (Millions)', axis=1).sum() / imdb['Revenue (Millions)'].sum() * 100

    genre_revenue_pcts[i] = revenue_pct

    i += 1

    print(genre, revenue_pct)
genre_revenue_pcts_df = pd.DataFrame(genre_revenue_pcts, index=individual_genres, columns=['Percent'])

genre_revenue_pcts_df
#Taking a sum of the revenue from the top 5 genres,

#which contributed to the most in terms of revenue?



explode = (0.05, 0.05, 0.08, 0.1, 0.12)

colors = ['#ff3232', '#ff4c4c', '#ff6666', '#ff7f7f', '#ff9999', ]

genre_revenue_pcts_df.sort_values(by='Percent', ascending=False).head(5).plot.pie(legend=False, subplots=True, autopct='%.2f%%', figsize=(8,8), colors=colors, explode=explode)

plt.ylabel('')

plt.title('Percent of Total Revenue from Top 5 Genres', weight='bold', fontsize=16)
#Find the most active directors, 'most active' being defined as number of films with their name on it

most_active_directors = imdb['Director'].value_counts().head(10)

most_active_directors.index
#Determine how much revenue each of these top 10 directors' films brought in 

#(as a sum for each director) in millions



director_revenue_totals = np.zeros(len(most_active_directors))

i = 0

for director in most_active_directors.index:

    current_director = imdb['Director'].str.contains(director).fillna(False)

    director_film_revenue = imdb[current_director].xs('Revenue (Millions)', axis=1).sum()

    director_revenue_totals[i] = director_film_revenue

    i += 1

    print(director, director_film_revenue)
director_revenue_totals_df = pd.DataFrame(director_revenue_totals, index=most_active_directors.index, columns=['Revenue (Millions)'])

director_revenue_totals_df
#Taking the sum of the revenue of the films made by the top 10 directors, which contributed the most?



explode = np.linspace(0, 0.5, 10)

colors = ['#ff0000', '#ff1919','#ff3232', '#ff4c4c', '#ff6666', '#ff7f7f', '#ff9999', '#ffb2b2', '#ffcccc', '#ffe5e5', ]

director_revenue_totals_df.sort_values(by='Revenue (Millions)', ascending=False).plot.pie(legend=False, subplots=True, autopct='%.2f%%', figsize=(8,8), colors=colors, explode=explode)

plt.ylabel('')

plt.title('Most Active Directors Revenue Contribution', weight='bold', fontsize=16)