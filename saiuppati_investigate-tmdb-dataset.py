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
# read in CSV data

movies = pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')

credits = pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')



# preview movies df

movies.head()
# preview credits df

credits.head()
# Use this cell to set up import statements for all of the packages that you

#   plan to use.

import matplotlib.pyplot as plt



# Remember to include a 'magic word' so that your visualizations are plotted

#   inline with the notebook.

%matplotlib inline
# Perform operations to inspect data types and look for instances of missing or possibly errant data

movies.info()
# english language movies check

movies.original_language.value_counts()
# Do we need to keep both original_title and title columns, let's see how many mismatches there are

(movies.original_title != movies.title).sum()
# let's check the values in status column

movies.status.value_counts()
# After discussing the structure of the data and any problems that need to be

#   cleaned, perform those cleaning steps in the second part of this section.



# drop unnecessary columns

cols_to_drop = ['original_title', 'homepage', 'id', 'overview', 'tagline', 'production_countries', 'spoken_languages']

movies.drop(cols_to_drop, axis=1, inplace=True)
# drop rows that are not originally english language movies

non_en_movies_ind = movies[movies.original_language != 'en'].index

movies.drop(non_en_movies_ind, inplace=True)

movies.original_language.value_counts()
# drop rows that don't have status 'Released'

unreleased_movies_ind = movies[movies.status != 'Released'].index

movies.drop(unreleased_movies_ind, inplace=True)

movies.status.value_counts()
# drop original_language and status columns

movies.drop(['original_language', 'status'], axis=1, inplace=True)

movies.info()
# drop missing value rows

movies.dropna(inplace=True)

movies.info()
# check how many movies have budget equal to 0

zero_budget = movies.budget == 0

print("{} movies with 0 budget in dataset".format(zero_budget.sum()))

movies[zero_budget]

# check how many movies have runtime equal to 0

(movies.runtime == 0).sum()
# drop rows that don't have 0 runtime

nonsense_runtime_ind = movies[movies.runtime == 0].index

movies.drop(nonsense_runtime_ind, inplace=True)
# movie genres columns, let's take a look at the first entry

movies.genres.iloc[0]
# generate list from entries in genres column

movie_genre_list = movies.genres.tolist()



# import module ast and convert string representation of list to actual list for each list of genres

import ast

movie_genre_list = [ast.literal_eval(lst) for lst in movie_genre_list]



# for each list of dictionaries, extract the genre strings and make list of strings

new_genre_list = []

for lst in movie_genre_list:

    movie_genres = [dct['name'] for dct in lst]

    new_genre_list.append(movie_genres)



# Overwrite 'genres' columns in dataframe with list of lists

movies['genres'] = new_genre_list



# verify the list of genres for first movie in dataframe

movies.genres.iloc[0]
# get the first entry of keywords column

movies.keywords.iloc[0]
# generate list from entries in keywords column

movie_keyword_list = movies.keywords.tolist()



# use module ast to convert string representation of list to actual list for each list of keywords

movie_keyword_list = [ast.literal_eval(lst) for lst in movie_keyword_list]



# for each list of dictionaries, extract the keyword strings and make list of strings

new_keyword_list = []

for lst in movie_keyword_list:

    movie_keyword = [dct['name'] for dct in lst]

    new_keyword_list.append(movie_keyword)



# Overwrite 'keywords' columns in dataframe with list of lists

movies['keywords'] = new_keyword_list



# verify the list of keywords for first movie in dataframe

movies.keywords.iloc[0]
# summary stats

movies.describe()
# create new movie dataframe with revenue/budget > 0

commercial_movie_filter = (movies.revenue > 0) & (movies.budget > 0)

commercial_movies = movies[commercial_movie_filter].copy()



# calculate profit

commercial_movies['profit'] = (commercial_movies.revenue - commercial_movies.budget) / commercial_movies.budget * 100



# view summary

commercial_movies.describe()
# distribution of profit values in a box-plot form

import seaborn as sns

sns.boxplot(x=commercial_movies.profit);

# commercial_movies.plot(kind='box')
# define upper hinge, lower hinge and IQR values

profit_lower_hinge = commercial_movies.profit.quantile(0.25)

profit_upper_hinge = commercial_movies.profit.quantile(0.75)

profit_IQR = profit_upper_hinge - profit_lower_hinge



# filter outliers

profit_outlier_filter = (commercial_movies.profit >= profit_lower_hinge - 1.5 * profit_IQR) & (commercial_movies.profit <= profit_upper_hinge + 1.5 * profit_IQR)

cm_filtered = commercial_movies[profit_outlier_filter].copy()



# see filtered values distribution

print(cm_filtered.describe())

sns.boxplot(x=cm_filtered.profit);
# scatter plot of profit v budget

ax = cm_filtered.plot(x='budget', y='profit', kind='scatter', figsize=(16,8))

ax.set_xlabel('Movie Budget, USD')

ax.set_ylabel('Profit Margin (%)')

ax.set_title('Relationship between movie profitability and budget');
# define mean profit margin for each class

lower_budget_profit_margin = cm_filtered.query('budget <= @cm_filtered.budget.mean()').profit.mean()

higher_budget_profit_margin = cm_filtered.query('budget > @cm_filtered.budget.mean()').profit.mean()

print('Lower budget movies have a mean profit margin of {}%.'.format(lower_budget_profit_margin))

print('Higher budget movies have a mean profit margin of {}%'.format(higher_budget_profit_margin))

fig, ax = plt.subplots()

ax.bar(['Lower', 'Higher'], [lower_budget_profit_margin, higher_budget_profit_margin])

ax.set_xlabel('Movie Budget Class')

ax.set_ylabel('Mean Profit Margin (%)');
# release date values

cm_filtered.release_date
# extract month from release_date

from datetime import datetime as dt

release_month = list(map(lambda x: dt.strptime(x, "%Y-%m-%d").month, cm_filtered.release_date))



# create release_month column in dataframe

cm_filtered['release_month'] = release_month
# distributions of profit margin by release month

fig, ax = plt.subplots(figsize=(16,8))

# cm_filtered.boxplot(column=['profit'], by=['release_month'], ax=ax)

sns.boxplot(x='release_month', y='profit', data=cm_filtered, ax=ax)

ax.set_xlabel('Movie Release Month')

ax.set_xticks(range(13))

ax.set_ylabel('Profit Margin (%)')

ax.set_title('Analysis of profit margin in relation to movie release month');
movies.popularity.hist();
movies.vote_average.hist();
movies.plot(x='vote_average', y='popularity', kind='scatter', figsize=(16,8));
# scatter plot of vote average and runtime

ax = movies.plot(x='runtime', y='vote_average', kind='scatter', figsize=(16,8))

ax.set_xlabel('Movie Runtime (min)')

ax.set_ylabel('Average Rating (10-point scale)')

ax.set_title('Relationship between movie runtime and rating');
# create rating category column

rating = ['Very poor' for i in range(len(cm_filtered))]

cm_filtered['rating'] = rating

cm_filtered.loc[((cm_filtered.vote_average > 2.5) & (cm_filtered.vote_average <= 5)), 'rating'] = 'Poor'

cm_filtered.loc[((cm_filtered.vote_average > 5) & (cm_filtered.vote_average <= 7.5)), 'rating'] = 'OK'

cm_filtered.loc[(cm_filtered.vote_average > 7.5), 'rating'] = 'Good'



# create a dictionary (4 elements) of dictionaries to keep track of counts of genres associated with each movie in the 4 rating categories

genres = {'Very poor': {}, 'Poor': {}, 'OK': {}, 'Good': {}}



# loop through and populate dictionaries of counts

for i in range(len(cm_filtered)):

    genre_list = cm_filtered.genres.iloc[i]

    rating = cm_filtered.rating.iloc[i]

    for genre in genre_list:

        if genre not in genres[rating]:

            genres[rating][genre] = 1

        else:

            genres[rating][genre] += 1
# sort to be able to get the top 3 genres in each rating category

genres_bar = {}

for key in genres:

    genres[key] = {k: v for k, v in sorted(genres[key].items(), key=lambda item: item[1], reverse=True)}



# visualization setup

n = 5

fig, axes = plt.subplots(2,2, figsize=(15,10))

sns.barplot(x=list(genres['Very poor'].keys())[:n], y=list(genres['Very poor'].values())[:n], ax=axes[0,0])

axes[0,0].set_yticks([0,1,2])

axes[0,0].set_ylabel('Count')

axes[0,0].set_title('Rating: Very poor')

sns.barplot(x=list(genres['Poor'].keys())[:n], y=list(genres['Poor'].values())[:n], ax=axes[0,1])

axes[0,1].set_ylabel('Count')

axes[0,1].set_title('Rating: Poor')

sns.barplot(x=list(genres['OK'].keys())[:n], y=list(genres['OK'].values())[:n], ax=axes[1,0])

axes[1,0].set_ylabel('Count')

axes[1,0].set_title('Rating: OK')

sns.barplot(x=list(genres['Good'].keys())[:n], y=list(genres['Good'].values())[:n], ax=axes[1,1])

axes[1,1].set_ylabel('Count')

axes[1,1].set_title('Rating: Good')

fig.suptitle('Top {} Genres in Each Movie Rating Category'.format(n));
# create profit margin category column

roi = ['Poor' for i in range(len(cm_filtered))]

cm_filtered['roi'] = roi

cm_filtered.loc[((cm_filtered.profit > 10) & (cm_filtered.profit <= 20)), 'roi'] = 'Acceptable'

cm_filtered.loc[((cm_filtered.profit > 20) & (cm_filtered.vote_average <= 50)), 'roi'] = 'Good'

cm_filtered.loc[(cm_filtered.profit > 50), 'roi'] = 'Excellent'



# create a dictionary (4 elements) of dictionaries to keep track of counts of genres associated with each movie in the 4 RoI categories

genres2 = {'Poor': {}, 'Acceptable': {}, 'Good': {}, 'Excellent': {}}



# loop through and populate dictionaries of counts

for i in range(len(cm_filtered)):

    genre_list = cm_filtered.genres.iloc[i]

    roi = cm_filtered.roi.iloc[i]

    for genre in genre_list:

        if genre not in genres2[roi]:

            genres2[roi][genre] = 1

        else:

            genres2[roi][genre] += 1



# sort to be able to get the top 3 genres in each RoI category

genres_bar = {}

for key in genres2:

    genres2[key] = {k: v for k, v in sorted(genres2[key].items(), key=lambda item: item[1], reverse=True)}



# visualization setup

n = 5

fig, axes = plt.subplots(2,2, figsize=(15,10))

sns.barplot(x=list(genres2['Poor'].keys())[:n], y=list(genres2['Poor'].values())[:n], ax=axes[0,0])

axes[0,0].set_yticks([0,1,2])

axes[0,0].set_ylabel('Count')

axes[0,0].set_title('RoI: Poor')

sns.barplot(x=list(genres2['Acceptable'].keys())[:n], y=list(genres2['Acceptable'].values())[:n], ax=axes[0,1])

axes[0,1].set_ylabel('Count')

axes[0,1].set_title('RoI: Acceptable')

sns.barplot(x=list(genres2['Good'].keys())[:n], y=list(genres2['Good'].values())[:n], ax=axes[1,0])

axes[1,0].set_ylabel('Count')

axes[1,0].set_title('RoI: Good')

sns.barplot(x=list(genres2['Excellent'].keys())[:n], y=list(genres2['Excellent'].values())[:n], ax=axes[1,1])

axes[1,1].set_ylabel('Count')

axes[1,1].set_title('RoI: Excellent')

fig.suptitle('Top {} Genres in Each RoI Category'.format(n));
# See how Marvel and DC movies are coded in keywords column

marvel_count = 0

dc_count = 0

for lst in cm_filtered.keywords:

    marvel = False

    dc = False

    for item in lst:

        if 'marvel' in item:

            print(item)

            if not marvel:

                marvel_count += 1

                marvel = True

        elif 'dc' in item:

            print(item)

            if not dc:

                dc_count += 1

                dc = True

print("Marvel count: ", marvel_count)

print("DC count: ", dc_count)
# create comic category column

comic = ['Other' for i in range(len(cm_filtered))]

cm_filtered['comic'] = comic

for i in range(len(cm_filtered)):

    if ('marvel comic' in cm_filtered.keywords.iloc[i]) or ('marvel cinematic universe' in cm_filtered.keywords.iloc[i]):

        cm_filtered.comic.iloc[i] = 'Marvel'

    elif ('dc comics' in cm_filtered.keywords.iloc[i]) or ('dc extended universe' in cm_filtered.keywords.iloc[i]):

        cm_filtered.comic.iloc[i] = 'DC'



# check how many of each type

cm_filtered.comic.value_counts()
# Marvel and DC movies rating and profit comparison

marvel_rating = cm_filtered[cm_filtered.comic == 'Marvel'].vote_average.mean()

marvel_profit = cm_filtered[cm_filtered.comic == 'Marvel'].profit.mean()

dc_rating = cm_filtered[cm_filtered.comic == 'DC'].vote_average.mean()

dc_profit = cm_filtered[cm_filtered.comic == 'DC'].profit.mean()



# Plot

labels = ['Marvel', 'DC']

ratings = [marvel_rating, dc_rating]

profits = [marvel_profit, dc_profit]

fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2, figsize=(10,5))

sns.barplot(labels, ratings, ax=ax1)

ax1.set_ylabel('Mean Rating')

sns.barplot(labels, profits, ax=ax2)

ax2.set_ylabel('Mean Profit Margin (%)')

fig.suptitle('Performance of Marvel vs DC Movies');
# Marvel and average movie rating and profit comparison

avg_rating = cm_filtered.vote_average.mean()

avg_profit = cm_filtered.profit.mean()



# Plot

labels = ['Marvel', 'Average']

ratings = [marvel_rating, avg_rating]

profits = [marvel_profit, avg_profit]

fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2, figsize=(10,5))

sns.barplot(labels, ratings, ax=ax1)

ax1.set_ylabel('Mean Rating')

sns.barplot(labels, profits, ax=ax2)

ax2.set_ylabel('Mean Profit Margin (%)')

fig.suptitle('Performance of Marvel vs Average Movie');