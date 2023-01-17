import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
# Your code here
chess= pd.Series([0.48, 0.44, 0.08], index=['white', 'black', 'draw'], name='winner')
print(chess)
# Your code here
chess_games.opening_name.map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()).value_counts() #counts how many times each opening_name occurs
# Your code here
chess_games.assign(
    victory_staus=chess_games.apply(lambda srs: srs.victory_status if pd.notnull(srs.victory_status) else srs.white_id, 
                        axis='columns')
)


chess_games.victory_status.value_counts() #counts how many times each piece of data occurs in the column
# Your code here
chessG=chess_games.assign(
    victory_staus=chess_games.apply(lambda srs: srs.victory_status if pd.notnull(srs.victory_status) else srs.white_id, 
                        axis='columns')
)
chessG.iloc[:20].pipe  #applies method chaing technique to the top 20 users
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
# Your code here
Kep=kepler.groupby(['koi_pdisposition', 'koi_disposition']).kepler_name.agg([len])
Kep.sort_index()
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
print(wine_reviews.head())
print(ramen_reviews.head())