%matplotlib inline



import matplotlib

import numpy as np

import pandas as pd

import seaborn as sb

import matplotlib.pyplot as plt
ratings_df = pd.read_csv('../input/recommendationmovieviacorrelation/movie_data_updated.tsv', sep='\t', encoding='ISO-8859-1', usecols=range(3), names=['userID', 'movieID', 'rating'])

titles_df = pd.read_csv('../input/recommendationmovieviacorrelation/movie_title_updated.csv', encoding='ISO-8859-1', usecols=range(2), names=['movieID', 'title'], skiprows=1)

ratings_df = pd.merge(ratings_df, titles_df, how='inner', on='movieID').reset_index(drop=True)

ratings_df.head()
count_df = ratings_df.groupby('userID')['movieID'].agg([np.size]).rename(columns={'size': 'review count'}).reset_index()

sb.set(style='whitegrid')

sb.boxplot(y=count_df['review count'])
count_df = count_df[count_df['review count'] < 320].reset_index(drop=True)

ratings_df = ratings_df[ratings_df['userID'].isin(count_df['userID'])]

pivot1 = ratings_df.pivot_table(index='userID', columns='title', values='rating')

pivot1.head()
times_a_movie_was_rated = pivot1.count(axis=0)

times_a_movie_was_rated.head()
sb.boxplot(y=times_a_movie_was_rated)

plt.show()
corelated = pivot1.corr(method='pearson', min_periods=159)

corelated.head()
Star_Wars_rating = corelated['Star Wars (1977)'].sort_values(ascending=False).dropna()

Star_Wars_rating.head()
user_ratings = ratings_df[ratings_df['userID'] == 0].set_index('title').rating

user_ratings.head()
recommendation = pd.Series(dtype=float)

for ind in user_ratings.index:

    suggestions = corelated[ind].dropna()

    if len(suggestions.index) != 0:

        if user_ratings[ind] >= 3:

            suggestions = suggestions.apply(lambda x: x * user_ratings[ind])

            recommendation = recommendation.append(suggestions)

        if user_ratings[ind] < 3:

            suggestions = suggestions.apply(lambda x: (-x) * user_ratings[ind])

            recommendation = recommendation.append(suggestions)



recommendation = recommendation.groupby(recommendation.index).sum().sort_values(ascending=False)

recommendation = recommendation[recommendation.index.isin(user_ratings)==False]



recommendation.head(10)