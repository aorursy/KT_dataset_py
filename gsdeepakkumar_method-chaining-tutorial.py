import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
# Your code here
chess_games['winner'].value_counts() / len(chess_games)
# Your code here
(chess_games
    .opening_name
    .map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip())
    .value_counts()
)
# Your code here
(chess_games[chess_games.winner=="white"].assign(n=0).groupby(['white_id','victory_status']).winner.apply(len).reset_index())
#answer_q3()
(chess_games
    .assign(n=0)
    .groupby(['white_id', 'victory_status'])
    .n
    .apply(len)
    .reset_index()
)
# Your code here
(chess_games.groupby('id').id.apply(len).sort_values(ascending=False).head(20))
#answer_q4()
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
(kepler.assign(n=0).groupby(['koi_pdisposition','koi_disposition']).n.apply(len).reset_index())
answer_q5()
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
wine_reviews.head()
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
ramen_reviews.head()
# Your code here
(((wine_reviews['points'].dropna() - 80) / 4)
    .value_counts()
    .sort_index()
    .rename_axis("Wine Ratings")
)
# Your code here
(ramen_reviews.Stars.replace("Unrated",None).dropna().astype('float64').value_counts().rename_axis('Ramen Ratings').sort_index())
# Your code here
(ramen_reviews.Stars.replace("Unrated",None).dropna().astype('float64').value_counts().map(lambda v: int(v) if v - int(v) < 0.5 else int(v) + 0.5).rename_axis('Ramen Ratings').sort_index())