import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 

sns.set_style("whitegrid")

%matplotlib inline



tmdb_df = pd.read_csv("../input/tmdbcsv/tmdb-movies.csv")

# Checking the data types and total number of data points before starting the analysis. 

tmdb_df.info()
tmdb_df.describe()
# Obtaining a list of genres

genre_details = list(map(str,(tmdb_df['genres'])))

genre = []

for i in genre_details:

	split_genre = list(map(str, i.split('|')))

	for j in split_genre:

		if j not in genre:

			genre.append(j)

# printing list of seperated genres.

print(genre)
# minimum range value

min_year = tmdb_df['release_year'].min()

# maximum range value

max_year = tmdb_df['release_year'].max()

# print the range

print(min_year, max_year)
# Creating a dataframe with genre as index and years as columns

genre_df = pd.DataFrame(index = genre, columns = range(min_year, max_year + 1))

# to fill not assigned values to zero

genre_df = genre_df.fillna(value = 0)

print (genre_df.head())
# list of years of each movie

year = np.array(tmdb_df['release_year'])

# index to access year value

z = 0

for i in genre_details:

    split_genre = list(map(str,i.split('|')))

    for j in split_genre:

        genre_df.loc[j, year[z]] = genre_df.loc[j, year[z]] + 1

    z+=1

genre_df
# number of movies in each genre so far.

genre_count = {}

genre = []

for i in genre_details:

	split_genre = list(map(str,i.split('|')))

	for j in split_genre:

		if j in genre:

			genre_count[j] = genre_count[j] + 1

		else:

			genre.append(j)

			genre_count[j] = 1

gen_series = pd.Series(genre_count)

# pi chart

gen_series = gen_series.sort_values(ascending = False)

label = list(map(str,gen_series[0:10].keys()))

label.append('Others')

gen = gen_series[0:10]

sum = 0

for i in gen_series[10:]:

    sum += i

gen['sum'] = sum

fig1, ax1 = plt.subplots()

ax1.pie(gen,labels = label, autopct = '%1.1f%%', startangle = 90)

ax1.axis('equal')

plt.title("Percentage of movies in each genre between 1960 and 2015")

plt.show()

# Creating a dataframe with genre as index and years as columns to get a count of popularity

popularity_df = pd.DataFrame(index = genre, columns = range(min_year, max_year + 1))

# to fill not assigned values to zero

popularity_df = popularity_df.fillna(value = 0.0)

print(popularity_df.head())
# list of popularity levels of each movie

popularity = np.array(tmdb_df['popularity'])

# to check whether any popularity is zero.

print (len(popularity[popularity==0]))

# index to access year value

z = 0

for i in genre_details:

    split_genre = list(map(str,i.split('|')))

    for j in split_genre:

            popularity_df.loc[j, year[z]] = popularity_df.loc[j, year[z]] + popularity[z]

    z+=1

popularity_df
# function to standardize the popularity of values in dataframe.

def standardize(p):

    p_std = (p - p.mean()) / p.std(ddof = 0)

    return p_std

popularity_std = standardize(popularity_df)

popularity_std
# Creating a series to hold the popular genre for every year.

pop_genre = pd.Series(index = range(min_year, max_year + 1))

pop_genre.head()
# to identify the genre with maximum standardized popularity value

for i in range(min_year, max_year + 1):

    pop_genre[i] = popularity_std[i].argmax()

pop_genre
# to plot a histogram of genre 'Drama'.

plt.plot(popularity_std.loc['Drama'])

plt.xlabel('year')

plt.ylabel('popularity levels')

plt.title('Distribution of popularity for the genre Drama over the years')

plt.axis([1960, 2015, 0, 3.5])

plt.show()

# Standardizing popularity data

std_pop = pd.Series(((popularity - popularity.mean()) / popularity.std(ddof = 0)), index = tmdb_df['id'])

std_pop.head()
# Obtaining budget data for positive values in standardized popularity.

budget = pd.Series(np.array(tmdb_df['budget']), index = tmdb_df['id'])

budget.head()
# to remove incomplete data from the dataset.

boolean = budget != 0

std_pop = std_pop[boolean]

budget = budget[boolean] 

budget.head()
print (budget.head(), std_pop.head())
print (len(std_pop), len(budget))
# Standardizing the budget values using the function standardize defined above

std_budget = standardize(budget)

std_budget.head()
# co relation coefficient(Pearson's value)

(std_pop * std_budget).mean()