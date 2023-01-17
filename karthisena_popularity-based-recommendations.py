import numpy as np

import pandas as pd



from IPython import display

from IPython.core.interactiveshell import InteractiveShell

import pprint

ratings_data = pd.read_csv('../input/ratings-small.csv')

movie_names = pd.read_csv('../input/movies-small.csv')
def show_two_heads(df1, df2, n=5):

    class A:

        def _repr_html_(self):

            return df1.head(n)._repr_html_() + '</br>' + df2.head(n)._repr_html_()

    return A()
show_two_heads(movie_names, ratings_data)
movie_data = pd.merge(ratings_data,movie_names,on='movieId')

movie_data.head()
movie_data.groupby('title')['rating'].mean().head()
movie_data.groupby('title')['rating'].count().sort_values(ascending=False).head()
ratings_mean_count = pd.DataFrame(movie_data.groupby('title')['rating'].mean())

ratings_mean_count.head()
ratings_mean_count['rating_counts'] = pd.DataFrame(movie_data.groupby('title')['rating'].count())



ratings_mean_count.sort_values(by='rating_counts',ascending=False)