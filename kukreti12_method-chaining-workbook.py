import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()


st =chess_games['winner'].value_counts() / len(chess_games)

check_q1(st)

ts =chess_games.opening_name.map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()).value_counts()
    
check_q2(ts)
ts= chess_games.assign(n=0).groupby(['white_id','victory_status']).n.apply(len).reset_index()
check_q3(ts)

ts= (chess_games
    .assign(n=0)
    .groupby(['white_id', 'victory_status'])
    .n
    .apply(len)
    .reset_index()
    .pipe(lambda df: df.loc[df.white_id.isin(chess_games.white_id.value_counts().head(20).index)])
)
ts
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
ts= (kepler
    .assign(n=0)
    .groupby(['koi_pdisposition','koi_disposition'])
    .n
    .count())
check_q5(ts)
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
print(wine_reviews.head())
print(ramen_reviews.head())
rat = ((wine_reviews.points.dropna() - 80) / 4).value_counts().sort_index()
rat.index.name = 'Wine Ratings'
rat.head()
check_q6(rat)
rating =ramen_reviews.Stars.replace('Unrated',None).dropna().astype('float64').value_counts().rename_axis('Ramen_Reviews').sort_index()
rating.head()
check_q7(rating)
def round_half(x):
    return round(x * 2) / 2
ramen_stars = (ramen_reviews.Stars[ramen_reviews.Stars.isin(["Unrated"]) == False]
               .astype("float64").map(lambda x: round_half(x)).value_counts()
               .sort_index().rename_axis("Ramen Reviews"))
ramen_stars