import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
# Your code here
chess_games['winner'].value_counts()/len(chess_games)
# note: value_counts sorts the frequency in order while groupby count sorts by lexicography
# Your code here
chess_games['opening_name'].map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()).value_counts()
# Your code here
#pd.DataFrame(chess_games.assign(n=0).groupby(['white_id', 'victory_status']).victory_status.count()).rename(columns={'':'Total_Numbers'})
chess_games.assign(n=0).groupby(['white_id', 'victory_status']).n.apply(len).reset_index()
# answer_q3()
# Your code here
def name_index(df):
    return df[df['n']>=20]
    df.index.name = 'review_id'
    return df
chess_games.assign(n=0).groupby(['white_id', 'victory_status']).n.apply(len).reset_index().pipe(name_index)
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
# Your code here
q5 = kepler.assign(n=0).groupby(['koi_pdisposition', 'koi_disposition']).n.count()
check_q5(q5)
q5
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
wine_reviews.head()
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
ramen_reviews.head()
# Your code here
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
q6=wine_reviews.assign(points = wine_reviews.apply(lambda srs: (srs.points - 80) /4, axis='columns')).groupby('points').points.count().rename("Wine Ratings")
check_q6(q6)
#answer_q6()
# alternative answer
(((wine_reviews['points'].dropna() - 80) / 4)
    .value_counts()
    .sort_index()
    .rename_axis("Wine Ratings")
)
q6
# Your code here
q7=ramen_reviews.Stars.replace('Unrated', None).dropna().astype('float64').value_counts().sort_index()
check_q7(q7)
q7
# Your code here
q8=ramen_reviews.Stars.replace('Unrated', None).dropna().astype('float64').map(lambda v: int(v) if v - int(v) < 0.5 else int(v) + 0.5).rename_axis("Ramen Reviews").value_counts().sort_index()
check_q8(q8)
q8