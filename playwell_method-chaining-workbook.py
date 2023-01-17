import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.opening_name.iloc[100:105]
# Your code here

check_q1(chess_games.winner.value_counts() / len(chess_games))

# Your code here


check_q2((chess_games
    .opening_name
    .map(lambda p: p.split(':')[0].split('|')[0].split('#')[0].strip())
    .value_counts()))


answer_q3()
a = (chess_games
    .assign(n=0)
    .groupby(['white_id', 'victory_status'])
    .n
    .count()
     .reset_index()
)
# Your code here
chess_games['n']=0

check_q3(chess_games.groupby(['white_id','victory_status']).n.count().reset_index())
cc = chess_games.white_id.value_counts().head(20).index

check_q4(a.pipe(lambda df:df.loc[df.white_id.isin(cc)]))

kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler

check_q5(kepler.assign(n=0).groupby(['koi_pdisposition', 'koi_disposition']).n.count())
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
print(wine_reviews.head())
print(ramen_reviews.head())
(((wine_reviews['points'] - 80) / 4)
    .value_counts()
    .sort_index()
     .rename_axis("Wine Ratings")
)
(((wine_reviews['points'].dropna() - 80) / 4)
    .value_counts()
    .sort_index()
    .rename_axis("Wine Ratings")
)
check_q7((ramen_reviews
    .Stars
    .replace('Unrated', None)
    .dropna()
    .astype('float64')
    .value_counts()
    .rename_axis("Ramen Reviews")
    .sort_index()))
check_q8((ramen_reviews
     .Stars
     .replace('Unrated', None)
     .dropna()
     .astype('float64')
     .map(lambda v: round(v * 2) /2)
     .value_counts()
     .rename_axis("Ramen Reviews")
     .sort_index()
))
check_q8((ramen_reviews
     .Stars
     .replace('Unrated', None)
     .dropna()
     .astype('float64')
     .map(lambda v: round(v * 2) / 2)
     .value_counts()
     .rename_axis("Ramen Reviews")
     .sort_index()
))