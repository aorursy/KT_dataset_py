import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
check_q1(chess_games.winner.value_counts() / len(chess_games))
help("Queen's Pawn Game: Zukertort Variation".split(":")[0].split("|")[0].split("#")[0].strip)
"Qu|een's Pawn Game: Zukertort Variation".split(":")[0].split("|")[0].split("#")[0].strip()
check_q2(chess_games
         .opening_name
         .map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip())
         .value_counts())
(chess_games
 .opening_name
 .map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip())
 .value_counts())
chess_games.groupby(['white_id', 'victory_status']).victory_status.value_counts()
check_q3(chess_games
         .assign(n='0')
         .groupby(['white_id', 'victory_status'])
         .n
         .apply(len)
         .reset_index())
(chess_games
    .assign(n='0')
    .groupby(['white_id', 'victory_status'])
    .n
    .apply(len)
    .reset_index())
chess_games.white_id.value_counts().head(20).index
check_q4(chess_games
         .assign(n='0')
         .groupby(['white_id', 'victory_status'])
         .n
         .apply(len)
         .reset_index()
         .pipe(lambda df: df.loc[df.white_id.isin(chess_games.white_id.value_counts().head(20).index)]))
check_q4(chess_games
         .assign(n='0')
         .groupby(['white_id', 'victory_status'])
         .n
         .count()
         .reset_index()
         .pipe(lambda df: df.loc[df.white_id.isin(chess_games.white_id.value_counts().head(20).index)]))
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
(kepler
 .assign(n='0')
 .groupby(['koi_pdisposition', 'koi_disposition'])
 .n
 .apply(len)
)
check_q5(kepler
 .assign(n='0')
 .groupby(['koi_pdisposition', 'koi_disposition'])
 .n
 .apply(len)
)
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
wine_reviews.head()
(wine_reviews.points.apply(lambda p: p / 5 - 15).rename_axis("Wine Ratings").sort_index())
# No missing data
(wine_reviews.points.dropna().apply(lambda p: p / 5 - 15).rename_axis("Wine Ratings").sort_index())
(wine_reviews.points
 .dropna().apply(lambda p: p / 5 - 15)
 .value_counts()
 .rename_axis("Wine Ratings")
 .sort_index())
answer_q6()
(((wine_reviews['points'].dropna() - 80) / 4)
    .value_counts()
    .sort_index()
    .rename_axis("Wine Ratings")
)
check_q6((((wine_reviews['points'].dropna() - 80) / 4)
    .value_counts()
    .sort_index()
    .rename_axis("Wine Ratings")
))
ramen_reviews.head()
(ramen_reviews
 .Stars
 .replace('Unrated', None)
 .dropna()
 .astype('float64')
 .value_counts()
 .rename_axis("Ramen ratings")
 .sort_index())
check_q7(ramen_reviews
 .Stars
 .replace('Unrated', None)
 .dropna()
 .astype('float64')
 .value_counts()
 .rename_axis("Ramen ratings")
 .sort_index()
)
answer_q8()
check_q8(ramen_reviews
 .Stars
 .replace('Unrated', None)
 .dropna()
 .astype('float64')
 .map(lambda v: round(v * 2) / 2)
 .value_counts()
 .rename_axis("Ramen ratings")
 .sort_index())