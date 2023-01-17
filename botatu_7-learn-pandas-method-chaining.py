import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
chess_games.winner.value_counts() / len(chess_games)
check_q1(chess_games.winner.value_counts()/len(chess_games))
chess_games.opening_name.map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()).value_counts()
check_q2(chess_games.opening_name.map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()).value_counts())
chess_games.groupby(['white_id', 'victory_status']).victory_status.count().reset_index(name='n')
check_q3(chess_games.groupby(['white_id', 'victory_status']).victory_status.count().reset_index(name='n'))
chess_games \
    .groupby(['white_id', 'victory_status']) \
    .victory_status \
    .count() \
    .reset_index(name='n') \
    .pipe(lambda x: x.loc[x.white_id.isin(chess_games.white_id.value_counts().head(20).index)])
check_q4(chess_games \
    .groupby(['white_id', 'victory_status']) \
    .victory_status \
    .count() \
    .reset_index(name='n') \
    .pipe(lambda x: x.loc[x.white_id.isin(chess_games.white_id.value_counts().head(20).index)]))
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
kepler.assign(n=0).groupby(['koi_pdisposition', 'koi_disposition']).n.count()
check_q5(kepler.assign(n=0).groupby(['koi_pdisposition', 'koi_disposition']).n.count())
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
display(wine_reviews.head())
display(ramen_reviews.head())
(((wine_reviews['points'].dropna() - 80) / 4)
    .value_counts()
    .sort_index()
    .rename_axis("Wine Ratings")
)
(ramen_reviews
     .loc[ramen_reviews.Stars != 'Unrated']
     .Stars
     .astype('float64')
     .value_counts()
     .sort_index()
     .rename_axis("Ramen Ratings")
)
(ramen_reviews
     .Stars
     .replace('Unrated', None)
     .dropna()
     .astype('float64')
     .map(lambda x: 0.5 * round(x / 0.5))
     .value_counts()
     .sort_index()
     .rename_axis("Ramen Ratings")
)