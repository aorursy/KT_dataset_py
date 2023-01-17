import pandas as pd 
import numpy as np
moviedata=pd.read_csv('../input/movies_metadata.csv')
moviedata.head()
C = moviedata['vote_average'].mean()
print(C)
m = moviedata['vote_count'].quantile(0.90)
print(m)
qualifiedmovies = moviedata.copy().loc[moviedata['vote_count'] >= m]
qualifiedmovies.shape
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)
    
qualifiedmovies['score'] = qualifiedmovies.apply(weighted_rating, axis=1)
qualifiedmovies = qualifiedmovies.sort_values('score', ascending=False)
qualifiedmovies[['title', 'vote_count', 'vote_average', 'score']].head(15)


