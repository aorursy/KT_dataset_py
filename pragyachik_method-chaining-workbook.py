import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
# Your code here
length=len(chess_games)
check_q1(chess_games.winner.value_counts()/length)
# answer_q1()
# Your code here
check_q2(chess_games['opening_name'].map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()).value_counts())
# Your code here
(chess_games
    .assign(n=0)
    .groupby(['white_id', 'victory_status'])
    .n
    .apply(len)
    .reset_index()
)
# Your code here
answer_q4()
(chess_games
    .assign(n=0)
    .groupby(['white_id', 'victory_status'])
    .n
    .apply(len)
    .reset_index()
    .pipe(lambda df: df.loc[df.white_id.isin(chess_games.white_id.value_counts().head(20).index)]) 
)
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
# Your code here
check_q5(kepler
     .assign(n=0)
    .groupby(['koi_pdisposition', 'koi_disposition'])
     .n
    .count()
)
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
print(wine_reviews.head())
print(ramen_reviews.head())
# Your code here
(((wine_reviews['points'].dropna() - 80) / 4)
    .value_counts()
    .sort_index()
    .rename_axis("Wine Ratings")
)
# Your code here
check_q7(ramen_reviews
    .Stars
     .replace('Unrated', None)
     .dropna()
     .astype('float64')            
     .value_counts()
     .rename_axis('Ramen Ratings')
     .sort_index()
)
# Your code here
check_q8(ramen_reviews
     .Stars
     .replace('Unrated', None)
     .dropna()
     .astype('float64')
     .map(lambda v: round(v * 2) / 2)
     .value_counts()
     .rename_axis("Ramen Reviews")
     .sort_index()
)