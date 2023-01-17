import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head(n=1)
# Your code here
count_of_all_games = len(chess_games)
check_q1((chess_games.winner.value_counts())/len(chess_games))

# Your code here
check_q2(chess_games.opening_name.map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()).value_counts())
#chess_games.opening_name.map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()).head()
#chess_games.opening_name
# Your code here


check_q3(chess_games
    .assign(n=0)
    .groupby(['white_id', 'victory_status'])
    .n
    .apply(len)
    .reset_index()
)
#print(answer_q3())
# Your code here
#check_q4(chess_games[["id", "turns"]].sort_values(by="turns", ascending=False).nlargest(columns = "turns", n = 20))

check_q4(chess_games
    .assign(n=0)
    .groupby(['white_id', 'victory_status'])
    .n
    .apply(len)
    .reset_index()
    .pipe(lambda df: df.loc[df.white_id.isin(chess_games.white_id.value_counts().head(20).index)]) 
)

chess_games.white_id.value_counts().head(20).index
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler.head(n=1)
# Your code here
check_q5(kepler.assign(n=0).groupby(["koi_pdisposition", "koi_disposition"]).n.count())
#print(answer_q5())
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
wine_reviews.head()
#print(ramen_reviews.head())
# Your code here
#pd.Series(wine_reviews.points/20, index = ["Wine Ratings"])
Wine_Ratings = (((wine_reviews['points'].dropna() - 80) / 4)
    .value_counts()
    .sort_index()
    .rename_axis("Wine Ratings")
)
print(Wine_Ratings)
# Your code here
df = ramen_reviews.copy()
check_q7(df.drop(df[df.Stars == "Unrated"].index).Stars.astype("float64").value_counts().sort_index().rename_axis("Ramen Ratings"))

# Your code here
#df.drop(df[df.Stars == "Unrated"].index).Stars.astype("float64").value_counts().sort_index().rename_axis("Ramen Ratings")ramen_reviews
check_q8(df
     .Stars
     .replace('Unrated', None)
     .dropna()
     .astype('float64')
     .map(lambda v: round(v * 2) / 2)
     .value_counts()
     .rename_axis("Ramen Reviews")
     .sort_index()
)



#import sqlite3
#connection = sqlite3.connect("../input/pitchfork-data/database.sqlite")
#dataframe = pd.read_sql_query("select * from reviews", connection)
#connection.close()
#dataframe.head()

#connection = sqlite3.connect("../input/european-football/database.sqlite")
#dataframe = pd.read_sql_query("select * from betfront", connection)
#connection.close()
#dataframe.head()

