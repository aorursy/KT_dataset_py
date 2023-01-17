import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
# Your code here
# also: chess_games.shape[0]
winner_per_rate = chess_games['winner'].value_counts() / len(chess_games)
# print(winner_per_rate)
check_q1(winner_per_rate)
# Your code here
opening_types = chess_games['opening_name'].map( lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()).value_counts()
# print(opening_types)
check_q2(opening_types)
# Your code here
winner_status_count = chess_games.assign(n=0).groupby(['white_id', 'victory_status'])['n'].apply(len).reset_index()
# print(winner_status_count)
check_q3(winner_status_count)
# Your code here
top_20_users = chess_games['white_id'].value_counts().head(20)
ans = winner_status_count.pipe(lambda df: df[df['white_id'].isin(top_20_users.index)])
# print(ans)
check_q4(ans)
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
# Your code here
pre_post_disposition = kepler.groupby(['koi_pdisposition', 'koi_disposition']).size()
# pre_post_disposition.plot()
check_q5(pre_post_disposition)
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
print(wine_reviews.head())
print(ramen_reviews.head())
# Your code here
points_normalize = (wine_reviews['points'] - 80) / 4
wine_ratings = points_normalize.value_counts().sort_index().rename('Wine Ratings')
# print(wine_ratings)
check_q6(wine_ratings)
# Your code here
convert_stars = ramen_reviews['Stars'].replace('Unrated', None).dropna().astype('float64')
ramen_ratings = convert_stars.rename('Ramen Ratings').value_counts().sort_index()
# print(ramen_ratings)
check_q7(ramen_ratings)
# Your code here
ramen_points = convert_stars.map(lambda v: int(v) if v-int(v)<0.5 else int(v)+0.5).rename('Ramen Points').value_counts().sort_index()
# print(ramen_points)
check_q8(ramen_points)