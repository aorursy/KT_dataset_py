import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
# create a srs whoing white win, black win, and draw
srs = (chess_games.winner
 .value_counts()/len(chess_games["winner"]))
check_q1(srs)
#answer_q1()
# 
srs = (chess_games
       .opening_name
       .map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip())
       .value_counts()
      )

check_q2(srs)
srs.nunique()
#answer_q2()
# Your code here

df = (chess_games
 .groupby(["white_id","victory_status"])
 .count()
)

"""
newdf = pd.DataFrame(chess_games
             .groupby(['white_id', 'victory_status'])
             .size()
             .rename('n')
             .reset_index()
            )
"""

#df = chess_games.assign(C=df['n'], D=df.B*2)
#new_df = chess_games.assign(n)
#print(n[10])
#check_q3(newdf)
# Your code here
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
# Your code here
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
print(wine_reviews.head())
print(ramen_reviews.head())
# Your code here
# Your code here
# Your code here