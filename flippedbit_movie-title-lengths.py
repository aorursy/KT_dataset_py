# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Python defacto plotting library

%matplotlib inline 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
movies = pd.read_csv('../input/movie_metadata.csv')

movies.shape
movies.head(n=2).T
movies.duplicated().sum()
movies[movies.duplicated()]['movie_title']
movies.drop_duplicates(inplace=True)

movies.shape
movies['letter_count'] = movies['movie_title'].str.len()

movies['letter_count'].describe()
title_lengths = movies['letter_count']

title_lengths.hist(bins=title_lengths.max(), log=True)

plt.title('Histogram of Title Letter Counts')

plt.xlabel('Number of Letters')

plt.ylabel('Frequency')
group_by_year = movies.groupby('title_year')

group_by_year['letter_count'].mean().plot(kind='line', figsize=(8,4))

plt.title('Average Number of Letters in Movies Titles by Year')

plt.ylabel('Average Number of Letters')

plt.xlabel('Year')
#http://stackoverflow.com/questions/34962104/pandas-how-can-i-use-the-apply-function-for-a-single-column

movies['word_count'] = movies['movie_title'].str.split().map(len)

movies[['movie_title', 'word_count', 'letter_count']].head()
# http://pandas.pydata.org/pandas-docs/version/0.15.0/visualization.html

movies.plot(kind='scatter', x='word_count', y='letter_count', xticks=range(16), figsize=(8,4), alpha=0.2)

plt.title('Movie Titles: Word Count vs. Letter Count')

plt.xlabel('Word Count')

plt.ylabel('Letter Count')