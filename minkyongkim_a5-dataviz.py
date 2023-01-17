# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import nltk

from nltk.corpus import brown



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
basicdf = pd.read_csv('/kaggle/input/imdbratings/title.basics.tsv', delimiter='\t', encoding='utf-8', header=0)

ratingdf = pd.read_csv('/kaggle/input/imdbratings/title.ratings.tsv', delimiter='\t', encoding='utf-8', header=0)

ratingdf = ratingdf.drop(['tconst'], axis=1)

# title = basicdf['primaryTitle'] # get the column 'primaryTitle' from the basics dataframe and assign it to variable title to get just the movie titles

# ratingdf['primaryTitle'] = title # then add the titles column to the ratings dataframe

combined = pd.concat([basicdf,ratingdf], axis=1)

mr = combined[combined.titleType == 'movie'] # keeps just the movies 

mr = mr.drop(columns=['tconst']) # drop the column with unique numerical identifiers

mr = mr.dropna() # drop all rows with NaN in them

mr = mr.replace(r'\\N',' ', regex=True) # replace \N character with whitespace 

sample_mr = mr.sample(n=200) # just print a sample of the actual dataset

sample_mr
rating = mr['averageRating']

votes = mr['numVotes']

plt.scatter(rating, votes) # scatterplot to see whether the number of votes given correlates with the average rating of movies.

plt.xlabel('Average Ratings Given')

plt.ylabel('Number of Votes Given')

plt.title('Correlation of Ratings and Votes') 

# no clear linear relationship between the ratings and number of votes given. In other words, despite a few number of votes, some movies received a very high rating, suggesting that each vote is weighted more than others. 
year = ['2019'] 

year_filtered = mr[mr.startYear.isin(year)] # filter the dataset for movies produced last year



six = year_filtered['genres'].value_counts().head(6) # value counts of genres show that most movies cluster into the top six genres, so just using these for the plot

six.plot(kind='bar') # plot the first six genres of movies produced last year

plt.xlabel('Movie Genres')

plt.ylabel('Number of Movies')

plt.title('Top Six Genres of Movies Produced in 2019')