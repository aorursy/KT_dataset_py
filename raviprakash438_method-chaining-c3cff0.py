import pandas as pd
pd.set_option('max_rows', 20)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
# Your code here
r1=chess_games.winner.value_counts()/len(chess_games)
check_q1(r1)
# Your code here
r2=chess_games.opening_name.map(lambda n: n.split(':')[0].split('|')[0].split('#')[0].strip()).value_counts()
check_q2(r2)
# Your code here
r3=chess_games.assign(n=0).groupby(['white_id','victory_status']).n.apply(len).reset_index()
check_q3(r3)
# Your code here
r3=chess_games.assign(n=0).groupby(['white_id','victory_status']).n.apply(len).reset_index()
usr=chess_games.white_id.value_counts().head(20).index
r4=r3[r3.white_id.isin(usr)]
check_q4(r4)
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
# Your code here
r5=kepler.assign(n=0).groupby(['koi_pdisposition', 'koi_disposition']).n.apply(len)
check_q5(r5)
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
wine_reviews.head()
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
ramen_reviews.head()
# Your code her
r6=((wine_reviews.points.dropna()-80)/4).value_counts().sort_index().rename('Wine Ratings')
check_q6(r6)
# Your code here
r7=ramen_reviews.drop(ramen_reviews[ramen_reviews.Stars=='Unrated'].index).Stars.astype('float64').value_counts().sort_index()
check_q7(r7)
ramen_reviews
# Your code here
r8=ramen_reviews.Stars.replace('Unrated',None).dropna().astype('float64').map(lambda x: int(x) if x - int(x)<0.5 else int(x)+0.5).value_counts().sort_index()
check_q8(r8)
