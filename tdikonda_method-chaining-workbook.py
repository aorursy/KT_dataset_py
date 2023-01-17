import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
df = chess_games.winner.value_counts()/chess_games.winner.count()
df
#check_q1(df)
df = chess_games.opening_name.map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()).value_counts()
df
#check_q2(df)
df = chess_games.groupby(['white_id', 'victory_status']).size().rename('n').reset_index()
df
#check_q3(df)
df = chess_games.white_id.value_counts().head(n=20)
df
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
kepler.assign(n=0).groupby(['koi_pdisposition', 'koi_disposition']).n.count()
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
print(wine_reviews.head())
print(ramen_reviews.head())
df = ((((wine_reviews['points'].dropna() - 80) / 5)+1).sort_index().rename_axis("Wine Ratings"))
df
df = ramen_reviews.Stars.replace('Unrated',None).dropna().astype('float64').value_counts().sort_index().rename_axis("Ramen Ratings")
df
check_q7(df)
ramen_reviews.head()
df = ramen_reviews.Stars.replace('Unrated',None).dropna().astype('float64').map(lambda s: round(s * 2) / 2).value_counts().sort_index().rename_axis("Ramen Ratings")
df
check_q8(df)