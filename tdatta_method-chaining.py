import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
# Your code here
(chess_games['winner']
    .value_counts()
    /len(chess_games))

# Your code here
(chess_games
    .opening_name
    .map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip())
    .value_counts()
)
# Your code here
(chess_games
    .assign(n=0)
    .groupby(['white_id', 'victory_status'])
    .n
    .apply(len)
    .reset_index()
)
# Your code here
(chess_games
    .assign(n=0)
    .groupby(['white_id'])
    .n
    .apply(len)
    .sort_values(ascending=False)
    .head(20)
    .reset_index())
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
# Your code here
kepler.assign(n=0).groupby(['koi_pdisposition', 'koi_disposition']).n.count()

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

#alternative approach 1
#wine_reviews['Wine Ratings'] = (wine_reviews['points'].dropna()-wine_reviews['points'].min()) / (wine_reviews['points'].max()-wine_reviews['points'].min())
#print(wine_reviews['Wine Ratings'].value_counts().sort_index())

#alternative approach 2
#wine_reviews['Wine Ratings'] = (wine_reviews['points'].dropna()-wine_reviews['points'].min()) / (wine_reviews['points'].max()-wine_reviews['points'].min())
#print(wine_reviews['Wine Ratings'].sort_values().sort_index())


# Your code here
(ramen_reviews['Stars']
    .replace('Unrated', None)
    .dropna()
    .astype('float64')
    .value_counts()
    .rename_axis("Ramen Ratings")
    .sort_index())
# Your code here
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