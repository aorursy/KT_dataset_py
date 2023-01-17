import pandas as pd
pd.set_option('max_rows', 10)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
'''Answer:
chess_games['winner'].value_counts() / len(chess_games)
'''
data = (chess_games.winner.value_counts()/chess_games.winner.count())
check_q1(data)
'''Answer:
(chess_games
    .opening_name
    .map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip())
    .value_counts()
)
'''
func = lambda txt: txt.split(':')[0].split('|')[0].split('#')[0].strip()
data = chess_games.opening_name.map(func).value_counts()
check_q2(data)
'''Answer:
(chess_games
    .assign(n=0)
    .groupby(['white_id', 'victory_status'])
    .n
    .apply(len)
    .reset_index()
)
'''
data = chess_games.assign(n=0).groupby(['white_id', 'victory_status']).n.count().reset_index()
check_q3(data)
''' Answer:
<Using method chaining>
(chess_games
    .assign(n=0)
    .groupby(['white_id', 'victory_status'])
    .n
    .apply(len)
    .reset_index()
    .pipe(lambda df: df.loc[df.white_id.isin(chess_games.white_id.value_counts().head(20).index)]) 
)
'''
'''
pd.df.pipe(func):
    將 df 當作參數，傳入 func 後，產生新的結果
'''
# <Doesn't use method chaining>
d1 = chess_games.white_id.value_counts().head(20).index # get the top20 users of playing games as white
d2 = chess_games.assign(n=0).groupby(['white_id', 'victory_status']).n.count().reset_index()
data = d2.loc[d2.white_id.isin(d1)]
check_q4(data)
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
data = kepler.assign(n=0).groupby(['koi_pdisposition', 'koi_disposition']).n.count()
check_q5(data)
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
wine_reviews.head(10)
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
ramen_reviews.head()
data = (
    wine_reviews
        .points
        .dropna()
        .map(lambda v: (v - 80)/ 4)
        .value_counts()
        .sort_index()
        .rename_axis(mapper='Wine Ratings')
)
check_q6(data)
''' Answer:
(ramen_reviews
    .Stars
    .replace('Unrated', None)
    .dropna()
    .astype('float64')
    .value_counts()
    .rename_axis("Ramen Reviews")
    .sort_index())
'''
data = (
ramen_reviews[ramen_reviews.Stars != 'Unrated']
    .Stars
    .astype('float64')
    .value_counts()
    .sort_index()
    .rename_axis('Ramen Ratings')
)
check_q7(data)
''' Answer:
(ramen_reviews
     .Stars
     .replace('Unrated', None)
     .dropna()
     .astype('float64')
     .map(lambda v: int(v) if v - int(v) < 0.5 else int(v) + 0.5)
     .value_counts()
     .rename_axis("Ramen Reviews")
     .sort_index()
)
'''
data = (
ramen_reviews[ramen_reviews.Stars != 'Unrated']
    .Stars
    .dropna()
    .astype('float64')
    .map(lambda v: int(v) if (v - int(v)) < 0.5 else int(v) + 0.5)
    .value_counts()
    .sort_index()
    .rename_axis('Ramen Reviews')
)
check_q8(data)