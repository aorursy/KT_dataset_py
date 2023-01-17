import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head(2)
q=chess_games.winner.value_counts().sum()
w=chess_games.winner.value_counts()
w.apply(lambda x: x/q)

open = chess_games.opening_name.map( lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()).value_counts()
open.head()
chess_games.assign(n=0).groupby(['white_id', 'victory_status']).n.apply(len).reset_index()
chess_games.assign(n=0).groupby(['white_id', 'victory_status']).n.apply(len).reset_index().pipe(lambda df: df.loc[df.white_id.isin(chess_games.white_id.value_counts().head(20).index)])
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
e = kepler.assign(n=1).groupby(['koi_pdisposition', 'koi_disposition']).n.count()
e.head()
check_q5(e)
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
print(wine_reviews.head())
print(ramen_reviews.head())
rat = ((wine_reviews.points.dropna() - 80) / 4).value_counts().sort_index()
rat.index.name = 'Wine Ratings'
rat.head()
check_q6(rat)
ratings = ramen_reviews.Stars.replace('Unrated', None).dropna().astype('float64').value_counts().rename_axis('Ramen Reviews').sort_index()
ratings.head()
check_q7(ratings)
ratings = ramen_reviews.Stars.replace('Unrated', None).dropna().astype('float64').map(lambda v: int(v) if v - int(v) < 0.5 else int(v) + 0.5).value_counts().rename_axis('Ramen Reviews').sort_index()
check_q8(ratings)