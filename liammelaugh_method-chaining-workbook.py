import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
# Your code here
chess_games.winner.value_counts()/len(chess_games)
check_q1(chess_games.winner.value_counts()/len(chess_games))
# Your code here
chess_games.opening_name.apply(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()).value_counts()
check_q2(chess_games.opening_name.apply(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()).value_counts())
# Your code here
chess_games["n"]=0
chess_games.groupby(["white_id","victory_status"]).n.value_counts().head()
chess_games.assign(n=0).groupby(["white_id","victory_status"]).n.apply(len).reset_index().n.value_counts().head()
# Your code here
check_q4(chess_games.assign(n=0).groupby(["white_id","victory_status"]).n.apply(len).reset_index().pipe(lambda df: df.loc[df.white_id.isin(chess_games.white_id.value_counts().head(20).index)]))
chess_games.white_id.value_counts().head(20).index
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
# Your code here
check_q5(kepler.assign(n=0).groupby(["koi_pdisposition", "koi_disposition"]).n.apply(len))
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
print(wine_reviews.head())
print(ramen_reviews.head())
# Your code here
wine_reviews.points.pipe(lambda srs: 1+(srs-80)/5).rename("Wine Ratings")
answer_q6()#For a start off this is a 0-5 scale and not what the question asked for. It said nothing 
#about counting the ratings and it says 80 is 1 star not 0 so mine is right yours is wrong
# Your code here
check_q7(ramen_reviews.Stars.replace("Unrated").astype("float64").value_counts().dropna().sort_index().rename("Ramen Ratings"))
round(.249*2)/2
# Your code here
check_q8(ramen_reviews.Stars.replace("Unrated").dropna().astype("float64").apply(lambda srs: round(srs*2)/2).value_counts().sort_index().rename("Ramen Ratings"))