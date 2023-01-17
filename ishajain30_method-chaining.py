import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from method_chaining import *

chess = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
# Your code here
a=chess.winner.value_counts()
b=a.sum()
a/b
# Your code here
chess.opening_name.apply(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()).value_counts()



(chess_games
    .opening_name
    .map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip())
    .value_counts()
)
chess.head()
# Your code here
chess.groupby('white_id')['victory_status'].value_counts()



(chess_games
    .assign(n=0)
    .groupby(['white_id', 'victory_status'])
    .n
    .apply(len)
    .reset_index()
)
# Your code here
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
# Your code here
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
wine_reviews.head()
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
ramen_reviews.head()
# Your code here
# Your code here
# Your code here