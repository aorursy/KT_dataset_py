import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()

ans1 = chess_games['winner'].value_counts() / len(chess_games)
check_q1(ans1)
# Your code here
ans2 = chess_games.opening_name.map( lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()).value_counts()
check_q2(ans2)

# Your code here
ans3 = (chess_games
         .assign(n=0)
         .groupby(['white_id', 'victory_status'])
         .n
         .apply(len)
         .reset_index()
       )
check_q3(ans3)
ans3
# Your code here
ans4 = (chess_games
        .assign(n=0)
        .groupby(['white_id', 'victory_status'])
        .n
        .apply(len)
        .reset_index()
        .pipe(lambda df: df.loc[df.white_id.isin(chess_games.white_id.value_counts().head(20).index)])
    )
check_q4(ans4)
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler


ans5 = kepler.assign(n=0).groupby(['koi_pdisposition', 'koi_disposition']).n.count()
check_q5(ans5)
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
wine_reviews.head()
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
ramen_reviews.head()

ans6 = (((wine_reviews['points'].dropna() - 80) / 4)
     .value_counts()
     .sort_index()
     .rename_axis("Wine Ratings")
)
check_q6(ans6)
# Your code here
ans7 = (ramen_reviews
        .Stars
        .replace('Unrated', None)
        .dropna()
        .astype('float64')
        .value_counts()
        .rename_axis("Ramen Reviews")
        .sort_index())
check_q7(ans7)
# Your code here
ans8 = (ramen_reviews
     .Stars
     .replace('Unrated', None)
     .dropna()
     .astype('float64')
     .map(lambda v: int(v) if v - int(v) < 0.5 else int(v) + 0.5)
     .value_counts()
     .rename_axis("Ramen Reviews")
     .sort_index()
)
check_q8(ans8)