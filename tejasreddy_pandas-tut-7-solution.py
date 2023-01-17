import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
res = chess_games["winner"].value_counts() / len(chess_games)
print(res)
print(check_q1(res))
res = chess_games.opening_name.map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()).value_counts()
print(res)
print(check_q2(res))
chess_games["n"]=0
res = chess_games.groupby(["white_id","victory_status"]).n.count().reset_index()
print(res)
print(check_q3(res))
# (chess_games
#     .assign(n=0)
#     .groupby(['white_id', 'victory_status'])
#     .n
#     .apply(len)
#     .reset_index()
# )
res
res = chess_games.assign(n=0).groupby(['white_id', 'victory_status']).n.apply(len).reset_index().pipe(lambda df: df.loc[df.white_id.isin(chess_games.white_id.value_counts().head(20).index)])
print(res)
print(check_q4(res))
res
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
res = kepler.groupby(["koi_pdisposition","koi_disposition"]).size()
print(res)
print(check_q5(res))
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
print(wine_reviews.head())
print(ramen_reviews.head())
res = wine_reviews.points.dropna().apply(lambda x: (x - 80)/4).value_counts().rename_axis("Wine Ratings").sort_index(ascending=True)
print(res)
print(check_q6(res))
res = ramen_reviews.Stars.replace("Unrated",None).dropna().astype("float64").value_counts().rename_axis("Ramen Reviews").sort_index(ascending=True)
print(res)
print(check_q7(res))
res = ramen_reviews.Stars.replace("Unrated",None).dropna().astype("float64").apply(lambda x: int(x) if (x-int(x))<0.5 else int(x)+0.5).value_counts().rename_axis("Ramen Reviews").sort_index(ascending=True)
print(res)
print(check_q8(res))