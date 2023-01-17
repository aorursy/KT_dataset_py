import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
q1 = chess_games['winner'].value_counts() / len(chess_games['winner'])
check_q1(q1)
q2 = chess_games['opening_name'].map(lambda n: n.split(':')[0].split('|')[0].split('#')[0].strip()).value_counts()
check_q2(q2)
q3 = chess_games.assign(n=0).groupby(["white_id", "victory_status"])["n"].count().reset_index()
check_q3(q3)
q4 = chess_games.assign(n=0).groupby(["white_id", "victory_status"])["n"].count().reset_index().pipe(lambda x: x.loc[x["white_id"].isin(chess_games["white_id"].value_counts().head(20).index)])
check_q4(q4)
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
q5 = kepler.assign(n=0).groupby(["koi_pdisposition", "koi_disposition"])["n"].count()
check_q5(q5)
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
print(wine_reviews.head())
print(ramen_reviews.head())
q6 = ((wine_reviews["points"].dropna() - 80) / 4).value_counts().sort_index().rename("Wine Ratings")
check_q6(q6)
q7 = ramen_reviews[ramen_reviews["Stars"] != "Unrated"]["Stars"].astype("float64").value_counts().rename("Ramen Ratings").sort_index()
check_q7(q7)
def round_half(x):
    return round(x * 2) / 2

q8 = ramen_reviews["Stars"].replace("Unrated", None).dropna().astype("float64").map(
    lambda x: round_half(x)).value_counts().rename_axis("Ramen Reviews").sort_index()
check_q8(q8)