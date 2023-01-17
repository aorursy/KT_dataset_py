import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
# Your code here
check_q1(chess_games.winner.value_counts() / chess_games.shape[0])
# Your code here
check_q2(chess_games.opening_name.map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()).value_counts())
# Your code here
check_q3(chess_games.assign(n=0).groupby(['white_id', 'victory_status']).n.apply(len).reset_index())
# Your code here
check_q4(chess_games.assign(n=0).groupby(['white_id', 'victory_status']).n.apply(len).reset_index().pipe(lambda df: df.loc[df.white_id.isin(chess_games.white_id.value_counts().head(20).index)]))
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
# Your code here
check_q5(kepler.assign(n=0).groupby(['koi_pdisposition', 'koi_disposition']).n.apply(len))
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
print(wine_reviews.head())
print(ramen_reviews.head())
# Your code here
check_q6(((wine_reviews['points'].dropna() - 80) / 4).value_counts().sort_index().rename_axis('Wine Ratings'))
# Your code here
check_q7(ramen_reviews.Stars.replace('Unrated', None).dropna().astype('float64').rename_axis('Ramen Ratings').value_counts().sort_index())
# Your code here
check_q8(ramen_reviews.Stars.replace('Unrated', None).dropna().astype('float64').map(lambda v: int(v) if v-int(v) < 0.5 else int(v) + 0.5).rename_axis('Ramen Reviews').value_counts().sort_index())
# while the value in replace is set to None, it will fill the value forward(default), but not fill it with NaN.
# In addition, if the first value will not be replaced though the condition is True
pd.set_option('max_rows', 0)
df = pd.DataFrame(data={'a': ['UN', '1', '2', 'UN', '3', '4', 'UN', '1'], 'b': [4,5,6,1,2,3,4,1]})
print(df.a.replace('UN', None)) # method = 'pad' as default
print(df.a.replace('UN', None, method='ffill'))
print(df.a.replace('UN', None, method='bfill'))