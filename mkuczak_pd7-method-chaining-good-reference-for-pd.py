import pandas as pd
pd.set_option('max_rows', 30)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
ans = (chess_games['winner']
       .value_counts()
       .map(lambda c: c/len(chess_games)))
check_q1(ans)
ans = (chess_games['opening_name']
       .map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip())
       .value_counts())
check_q2(ans)
ans = (chess_games
       .assign(n=0)
       .groupby(['white_id','victory_status'])
       .n
       .apply(len)
       .reset_index()) # This refers to the len of this grouping (white_id with a particular victory_status).
check_q3(ans)
ans = (chess_games
       .assign(n=0)
       .groupby(['white_id', 'victory_status'])
       .n
       .apply(len)
       .reset_index()
       .pipe(lambda df: df.loc[df.white_id.isin(chess_games.white_id.value_counts().head(20).index)]))
check_q4(ans)
# You're looking at just the white_id column (you don't care how many games they played as black, apparently)
# Then you are only displaying the rows from chess_games that 'isin' the top 20 white_id.
# chess_games.white_id.value_counts().head(20).index grabs only the top 20 players.
# df.loc[df.white_id.isin(**ABOVE**)] scours the white_id col in the original database for the top 20 and displays the full row.
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
ans = (kepler
       .assign(n=0)
       .groupby(['koi_pdisposition', 'koi_disposition'])
       .n
       .count()) # apply(len) should work, too
check_q5(ans)
# The values on the graph make sense.  Unlikely that you'd say it is a false positive and then later confirm.
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
wine_reviews.head()
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
ramen_reviews.head()
ans = (wine_reviews['points']
       .map(lambda p: (p-80)/4) # apparently you can just write (wine_reviews['points'] - 80) / 4
       .value_counts()
       .rename_axis('Wine Ratings')
       .sort_index()
      )
check_q6(ans)
ans = (ramen_reviews['Stars']
       .replace('Unrated', None)
       .dropna()
       .astype('float64')
       .value_counts()
       .sort_index()
       .rename_axis('Ramen Ratings')
      )
check_q7(ans)
mod_ans = (ramen_reviews['Stars']
           .replace('Unrated', None)
           .dropna()
           .astype('float64')
           .map(lambda v: int(v) if v - int(v) < 0.5 else int(v) + 0.5
               )
           .value_counts()
           .rename_axis('Ramen Reviews')
           .sort_index())
check_q8(mod_ans)