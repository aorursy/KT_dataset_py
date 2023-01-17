import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
# Your code here
total = len(chess_games.id)
total

chess_games['winner'].value_counts()/total

check_q1(chess_games['winner'].value_counts()/total)
# Your code here
chess_games.head()
opening_archetypes = chess_games.opening_name.map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip())
opening_archetypes.value_counts()
check_q2(opening_archetypes.value_counts())
# Your code here
chess_games.head()
chess_games.assign(n=0).groupby(['white_id','victory_status']).n.count().reset_index()
check_q3(chess_games.assign(n=0).groupby(['white_id','victory_status']).n.count().reset_index())
# Your code here
df = chess_games.assign(n=0).groupby(['white_id','victory_status']).n.count().reset_index()
df2 = chess_games.white_id.value_counts().head(20)
df[df.white_id.isin(df2.index)]
check_q4(df[df.white_id.isin(df2.index)])
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
# Your code here
kepler.groupby(['koi_pdisposition','koi_disposition']).kepid.count()
check_q5(kepler.groupby(['koi_pdisposition','koi_disposition']).kepid.count())
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
wine_reviews.head()
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
ramen_reviews.head()
# Your code here
(((wine_reviews['points'].dropna() - 80)/4).value_counts().sort_index().rename_axis('Wine Ratings'))
check_q6((((wine_reviews['points'].dropna() - 80)/4).value_counts().sort_index().rename_axis('Wine Ratings')))
# Your code here
# change to float type
ramen_reviews.Stars = ramen_reviews.Stars.replace('Unrated',None).dropna().astype('float64')
# summary
ramen_reviews.Stars.value_counts().rename_axis('Ramen Ratings').sort_index()
check_q7(ramen_reviews.Stars.value_counts().rename_axis('Ramen Ratings').sort_index())
# Your code here
a = (ramen_reviews
    .Stars
    .replace('Unrated',None)
    .dropna()
    .astype('float64')
    .map(lambda v: int(v) if v - int(v) < 0.5 else int(v) + 0.5)
    .value_counts()
    .rename_axis('Ramen Reviews')
    .sort_index()
)
check_q8(a)
