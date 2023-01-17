import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
# Your code here
res = chess_games["winner"].value_counts() / len(chess_games)
print(res)
print(check_q1(res))

# Your code here
opening_archetypes = chess_games.opening_name.map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip())
res = opening_archetypes.value_counts()

print(check_q2(res))

opening_archetypes = chess_games.opening_name.map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip())
opening_archetypes.value_counts()
check_q2(opening_archetypes.value_counts())

# Your code here
res = chess_games.assign(n=0).groupby(['white_id','victory_status']).n.count().reset_index()
print(res)

print(check_q3(res))

# Your code here
df = chess_games.assign(n=0).groupby(['white_id','victory_status']).n.count().reset_index()
df2 = chess_games.white_id.value_counts().head(20)
res = df[df.white_id.isin(df2.index)]
print(res)
print(check_q4(res))

kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
# Your code here
res = kepler.groupby(['koi_pdisposition', 'koi_disposition']).kepid.count()
print(res)
check_q5(res)
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
print(wine_reviews.head())
print(ramen_reviews.head())
# Your code here
res = (((wine_reviews['points'].dropna() - 80)/4).value_counts().sort_index().rename_axis('Wine Ratings'))
print(res)
check_q6(res)
check_q6((((wine_reviews['points'].dropna() - 80)/4).value_counts().sort_index().rename_axis('Wine Ratings')))
# Your code here
ramen_reviews.Stars = ramen_reviews.Stars.replace('Unrated', None).dropna().astype('float64')

ramen_reviews.Stars.value_counts().rename_axis('Ramen Ratings').sort_index()
check_q7(ramen_reviews.Stars.value_counts().rename_axis('Ramen Ratings').sort_index())

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
