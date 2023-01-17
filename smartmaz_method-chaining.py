import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
# Your code here
d1=chess_games['winner'].value_counts() / len(chess_games)
check_q1(d1)
# Your code here
d2=chess_games.opening_name.map(lambda n: n.split(':')[0].split('|')[0].split('#')[0].strip()).value_counts()
check_q2(d2)
# Your code here
d3=chess_games.assign(n=0).groupby(['white_id', 'victory_status']).n.apply(len).reset_index()
check_q3(d3)
# Your code here
d4=(chess_games.assign(n=0).groupby(['white_id', 'victory_status']).n.apply(len).reset_index()
.pipe(lambda df: df.loc[df.white_id.isin(chess_games.white_id.value_counts().head(20).index)]))
check_q4(d4)
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
# Your code here
d5=(kepler.groupby(['koi_pdisposition', 'koi_disposition'])).rowid.count()
check_q5(d5)
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
wine_reviews.head()
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
ramen_reviews.head()
# Your code here
d6 = ((wine_reviews['points'].dropna()-80) / 4).value_counts().sort_index().rename_axis('Wine Ratings')
check_q6(d6)
# Your code here
d7=ramen_reviews['Stars'].replace('Unrated', None).dropna().astype('float64').value_counts().sort_index().rename_axis("Ramen Reviews")
check_q7(d7)
# Your code here
d8=(ramen_reviews['Stars'].replace('Unrated', None).dropna().astype('float64').
map(lambda d: int(d) if d - int(d) < 0.5 else int(d) + 0.5).value_counts().sort_index().rename_axis("Ramen Reviews"))
check_q8(d8)