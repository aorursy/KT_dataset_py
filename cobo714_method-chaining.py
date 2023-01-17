import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
chess_games.winner.value_counts() / len(chess_games)
chess_games.opening_name.map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()).value_counts()
chess_games.assign(n=0).groupby(["white_id", "victory_status"]).n.count().reset_index()
check_q3(chess_games.assign(n=0).groupby(["white_id", "victory_status"]).n.count().reset_index())
chess_games.assign(n=0).groupby(["white_id", "victory_status"]).n.count().reset_index().pipe(lambda df: df.loc[df.white_id.isin(chess_games.white_id.value_counts().head(20).index)])
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
transitions = kepler.groupby(["koi_pdisposition", "koi_disposition"]).rowid.count()
check_q5(transitions)
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
wine_reviews.head()
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
ramen_reviews.head()
wine_ratings = (((wine_reviews.points.dropna() - 80) / 4)).value_counts().sort_index().rename("Wine Ratings")
check_q6(wine_ratings)

# Solution provided by answer_q6() and verified by check_q6() seems to be incorrect
# wine_ratings = (((wine_reviews.points.dropna() - 80) / 5) + 1).value_counts().sort_index().rename("Wine Ratings")
ramen_ratings = ramen_reviews.Stars.replace("Unrated", None).dropna().astype("float64").value_counts().rename("Ramen Ratings").sort_index()
#ramen_ratings = ramen_reviews.Stars[ramen_reviews.Stars != "Unrated"].astype("float64").value_counts().rename("Ramen Ratings").sort_index()
check_q7(ramen_ratings)
ramen_ratings_round = ramen_reviews.Stars.replace("Unrated", None).dropna().astype("float64").map(lambda v: int(v) if v - int(v) < 0.5 else int(v) + 0.5).value_counts().rename("Ramen Reviews").sort_index()
check_q8(ramen_ratings_round)

# Solution provided here is also incorrect due to its lambda function rounding the points incorrectly.  It actually floors to the nearest half-point rather than rounding as described in the exercise
# ramen_ratings_round = ramen_reviews.Stars.replace("Unrated", None).dropna().astype("float64").map(lambda r: round(r / 0.5) * 0.5).value_counts().rename("Ramen Ratings").sort_index()