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
# Your code here
opening_archetype = chess_games.opening_name.map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip())
opening_archetype.value_counts()
answer_q2()
# Your code here
chess_games["n"] = 0
white_vic = chess_games.groupby(["white_id", "victory_status"]).n.apply(len).reset_index()
# Your code here

idx = chess_games.white_id.value_counts().head(20).index
check_q4(white_vic[white_vic["white_id"].isin(idx)])
#white_vic.pipe(lambda df: df.loc[df.white_id.isin(chess_games.white_id.value_counts().head(20).index)])
#answer_q4()
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler.sample(5)
# Your code here
check_q5(kepler.assign(n=0).groupby(["koi_pdisposition", "koi_disposition"]).n.count())
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
wine_reviews.head()
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
ramen_reviews.head()
# note: the prompt is a bit misleading as it says that 1 star is the minimum (1 to 5 point scale as opposed to 0)
#this particular code rates from 0 to 5.
wine_ratings = pd.Series(((wine_reviews.points-80)/4),name="Wine Ratings").value_counts().sort_index()
check_q6(wine_ratings)
#check_q6(((wine_reviews.points.dropna()-80)/4).value_counts().sort_index())
#answer_q6()
# Your code here
ramen_stars = (ramen_reviews.Stars[ramen_reviews.Stars.isin(["Unrated"]) == False].astype("float64").
 value_counts().sort_index().rename_axis("Ramen Reviews"))
check_q7(ramen_stars)
# Your code here
#the rounding method is different. The one provided in the answer is essentially "floor" method rounded to 0.5
def round_half(x):
    return round(x * 2) / 2
ramen_stars = (ramen_reviews.Stars[ramen_reviews.Stars.isin(["Unrated"]) == False]
               .astype("float64").map(lambda x: round_half(x)).value_counts()
               .sort_index().rename_axis("Ramen Reviews"))
ramen_stars
#check_q8(ramen_stars)
check_q8((ramen_reviews
     .Stars
     .replace('Unrated', None)
     .dropna()
     .astype('float64')
     .map(lambda v: int(v) if v - int(v) < 0.5 else int(v) + 0.5)
     .value_counts()
     .rename_axis("Ramen Reviews")
     .sort_index()
))
