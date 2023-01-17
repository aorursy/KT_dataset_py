import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
chess_games.winner.value_counts().map(lambda x: x/len(chess_games))
check_q1(chess_games.winner.value_counts().map(lambda x: x/len(chess_games)))
chess_games.assign(
    opening_name=chess_games.apply(lambda srs: srs.opening_name.split(":")[0].split("|")[0].split("#")[0].strip(), 
                        axis='columns')
).opening_name.value_counts()
check_q2(chess_games.assign(
    opening_name=chess_games.apply(lambda srs: srs.opening_name.split(":")[0].split("|")[0].split("#")[0].strip(), 
                        axis='columns')
).opening_name.value_counts())
chess_games.groupby(['white_id', 'victory_status']).size().reset_index(name='n')
check_q3(chess_games.groupby(['white_id', 'victory_status']).size().reset_index(name='n'))
(chess_games
    .groupby(['white_id', 'victory_status'])
    .size().reset_index(name='n')
    .pipe(lambda df: df[df.white_id.isin(chess_games.white_id.value_counts().head(20).index)])
)
check_q4((chess_games
    .groupby(['white_id', 'victory_status'])
    .size().reset_index(name='n')
    .pipe(lambda df: df[df.white_id.isin(chess_games.white_id.value_counts().head(20).index)])
))
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
kepler.groupby(['koi_pdisposition', 'koi_disposition']).size()
check_q5(kepler.groupby(['koi_pdisposition', 'koi_disposition']).size())
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
print(wine_reviews.head())
print(ramen_reviews.head())
(wine_reviews.points
     .map(lambda x: (x-80)/20*5)
     .to_frame(name='Wine Ratings')
     .groupby('Wine Ratings').size()
)
check_q6((wine_reviews.points
     .map(lambda x: (x-80)/20*5)
     .to_frame(name='Wine Ratings')
     .groupby('Wine Ratings').size()
))
(ramen_reviews.Stars
    .replace('Unrated', None)
    .dropna()
    .astype('float64')
    .value_counts()
    .rename('Ramen Ratings')
    .sort_index()
)
check_q7((ramen_reviews.Stars
    .replace('Unrated', None)
    .dropna()
    .astype('float64')
    .value_counts()
    .rename('Ramen Ratings')
    .sort_index()
))
(ramen_reviews.Stars
    .replace('Unrated', None)
    .dropna()
    .astype('float64')
    .map(lambda x: round(x * 2) / 2)
    .value_counts()
    .rename('Ramen Ratings')
    .sort_index()
)
check_q8((ramen_reviews.Stars
    .replace('Unrated', None)
    .dropna()
    .astype('float64')
    .map(lambda x: round(x * 2) / 2)
    .value_counts()
    .rename('Ramen Ratings')
    .sort_index()
))