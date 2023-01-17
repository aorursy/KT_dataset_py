import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np
df_movie = pd.read_csv("../input/tmdb_5000_movies.csv")

df_credit = pd.read_csv("../input/tmdb_5000_credits.csv")
# Since genres is given in dictionary format, we have to extract first value from each genres

df_movie.genres = df_movie.genres.str.split(",",n=2,expand=True)[1]

df_movie.genres
# Check for any nulls after splitting it:

df_movie.genres.isnull().sum()
df_movie.genres.dropna(inplace=True)
# Confirm all nulls are filled with 0

df_movie.genres.isnull().sum()
df_movie['genres'] = df_movie['genres'].apply(lambda x: x.split('"name"')[1])
df_movie['genres']
# Remove all special character from genres

df_movie['genres'] =  df_movie['genres'].str.replace('"', '')

df_movie['genres'] =  df_movie['genres'].str.replace(':', '')

df_movie['genres'] =  df_movie['genres'].str.replace(']', '')

df_movie['genres'] =  df_movie['genres'].str.replace('}', '')
# Remove white space from genres

df_movie['genres'] = df_movie['genres'].str.strip()
# Confirm all the values are proper

df_movie['genres']
# Extract year from release date

df_movie['release_date'] = pd.to_datetime(df_movie['release_date'])
df_movie['year'] = df_movie['release_date'].dt.year
df_movie.year = df_movie.year.fillna(0).astype(int)
# Confirm that new column year has proper values

df_movie.year
# Highest top 3 categories 

df_movie.groupby('genres').revenue.sum().nlargest(3)
# For category- Action, plot the last 10 years moving average trend

df_action = df_movie.query('genres == "Action"').groupby('year').revenue.sum()

df_action = df_action.rolling(window=10).mean()

df_action 
df_action.dropna(inplace=True)

plt.subplots(figsize=(10, 6))

plt.bar(df_action.index, df_action)

plt.title('10 Years moving average for category - Action')

plt.xlabel('Year')

plt.ylabel('Moving average');
# For category- Adventure, plot the last 10 years moving average trend

df_adventure = df_movie.query('genres == "Adventure"').groupby('year').revenue.sum()

df_adventure = df_adventure.rolling(window=10).mean()

df_adventure 
df_adventure .dropna(inplace=True)

plt.subplots(figsize=(10, 6))

plt.bar(df_adventure .index, df_adventure )

plt.title('10 Years moving average for category Adventure')

plt.xlabel('Year')

plt.ylabel('Moving average');
# For category- Drama, plot the last 10 years moving average trend

df_drama = df_movie.query('genres == "Drama"').groupby('year').revenue.sum()

df_drama = df_drama.rolling(window=10).mean()

df_drama
df_drama.dropna(inplace=True)

plt.subplots(figsize=(10, 6))

plt.bar(df_drama.index, df_drama)

plt.title('10 Years moving average for category Drama')

plt.xlabel('Year')

plt.ylabel('Moving average');
df_movie.plot.scatter(x='revenue',y='budget');

df_movie.plot.scatter(x='revenue',y='popularity');

df_movie.plot.scatter(x='revenue',y='runtime');

df_movie.plot.scatter(x='revenue',y='vote_average');
# To analyse production companies, we have to first clean the dataset for production comapanies.

df_movie.production_companies = df_movie.production_companies.str.split(":",n=2,expand=True)[1]
df_movie['production_companies'] =df_movie['production_companies'].fillna(0)
df_movie['production_companies'] = df_movie['production_companies'].str.split(",",n=1,expand=True)[0]
df_movie['production_companies'] = df_movie['production_companies'].str.replace('"', '')

df_movie['production_companies'] = df_movie['production_companies'].str.strip()
# Confirm all changes done properly

df_movie['production_companies']
# creating a subset from larger dataset

df_movie_small = df_movie[['year','production_companies','revenue']] 
# Find out the maximum revnue generating production company from each year

df_new = df_movie_small.ix[df_movie_small.groupby(['year']).revenue.idxmax()]
df_new
# Count of years in which production company has generated maximum revennue

df_new = df_new.groupby('production_companies').year.count()
# To visualise we have taken where production company has highest grosser more than 1 year

df_new = df_new[df_new > 1]

df_new
df_new.plot(kind='bar', stacked=True, colormap='autumn')
# Now we look into credits dataset to analyse some pattern

df_credit
# To start with we have to clean the dataset first, in particular cast column

df_credit['cast'] = df_credit['cast'].str.split(":",n=6,expand=True)[6]

df_credit['cast'] = df_credit['cast'].str.split(",",n=0,expand=True)[0]

df_credit['cast']
df_credit['cast'] =  df_credit['cast'].str.replace('"', '')

df_credit['cast'] = df_credit['cast'].str.strip()
# Merge both the credit and movie dataset

df_combined = df_credit.merge(df_movie, left_on='movie_id', right_on='id', how='inner')
df_cast = df_combined[['cast','year','revenue']]
df_cast = df_cast.ix[df_cast.groupby(['year']).revenue.idxmax()]
df_cast
df_cast = df_cast.groupby('cast').year.count()

df_cast  = df_cast [df_cast >1]

df_cast
df_cast.plot(kind='bar', stacked=True, colormap='autumn')