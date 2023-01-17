import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
# Your code here
n =  len(chess_games)
chess_games.winner.value_counts()/n
check_q1(chess_games.winner.value_counts()/n)
# Your code here
f = lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()
opening_names = chess_games.opening_name.map(f)

check_q2(opening_names.value_counts())

# Your code here
chess_games['n'] = 0
check_q3(chess_games.groupby(['white_id','victory_status']).n.count().reset_index())
print(chess_games.groupby(['white_id','victory_status']).n.count().reset_index())
#chess_games.assign(n=0).groupby(['white_id', 'victory_status']).n.apply(len).reset_index()
# Your code here
def f(df) : 
    df['n'] = 0
    tmp = df.groupby(['white_id','victory_status']).n.count().reset_index()
    return tmp.loc[df.n>= 20, 'n']
chess_games.pipe(f)
answer_q4()
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