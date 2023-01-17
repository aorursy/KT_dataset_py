import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
chess_games['winner'].value_counts() / chess_games['winner'].value_counts().sum()
answer_q1()
check_q1(chess_games['winner'].value_counts() / chess_games['winner'].value_counts().sum())
chess_games['opening_name'].map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()).value_counts()
chess_games['n'] = 0
chess_games.groupby(['white_id', 'victory_status'])['n'].count().reset_index()
chess_games['n'] = 0
answer = chess_games.groupby(['white_id', 'victory_status'])['n'].count().reset_index()
answer = answer.loc[answer.white_id.isin(chess_games.white_id.value_counts().head(20).index)]
check_q4(answer)
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
kepler['n'] = 0
kepler.groupby(['koi_pdisposition', 'koi_disposition'])['n'].value_counts()

wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
print(wine_reviews.head())
print(ramen_reviews.head())
answer_q6()
# Your code here
# Your code here