import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
check_q1((chess_games.winner.value_counts()) / (chess_games.winner.count()))
check_q2((chess_games.opening_name.apply( lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip())).value_counts())
chess_games['n'] = 0
kk = chess_games.groupby(['white_id', 'victory_status']).n.apply(len).reset_index()
# kk taken from previous exercise

idx = chess_games.white_id.value_counts().head(20).index
asd = kk[kk.white_id.isin(idx)]

print(asd)

check_q4(asd)
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
kepler['n']=0
k = kepler.groupby(['koi_pdisposition', 'koi_disposition']).n.count()
print(k)
check_q5(k)
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
wine_reviews.head()
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
ramen_reviews.head()
rating = pd.Series(((wine_reviews.points-80)/4),name="Wine Ratings").value_counts().sort_index()
print(rating)
check_q6(rating)
ramen_reviews.head(3)
star = (ramen_reviews.Stars[ramen_reviews.Stars.isin(["Unrated"]) == False].astype("float64"). \
 value_counts().sort_index().rename_axis("Ramen Reviews"))
print(pd.DataFrame(star))
check_q7(star)
# Another method to do Exercise 8
# The rounding method is different. The one provided in the answer is essentially "floor" method rounded to 0.5
def round_half(x):
    return round(x * 2) / 2
ramen_stars = (ramen_reviews.Stars[ramen_reviews.Stars.isin(["Unrated"]) == False]
               .astype("float64").map(lambda x: round_half(x)).value_counts()
               .sort_index().rename_axis("Ramen Reviews"))
q = ramen_reviews.Stars.replace('Unrated', None).dropna().astype('float64').map(lambda v: int(v) if v - int(v) < 0.5 else int(v) + 0.5)\
.value_counts().rename_axis("Ramen Reviews").sort_index()

print(q)

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