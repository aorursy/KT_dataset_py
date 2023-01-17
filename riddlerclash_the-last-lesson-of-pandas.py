import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
# Your code here
white = len(chess_games.loc[chess_games.winner == 'white'])
black = len(chess_games.loc[chess_games.winner == 'black'])
draw = len(chess_games.loc[chess_games.winner == 'draw'])
total = len(chess_games.winner)
pd.Series([white/total, black/total, draw/total], index=['white', 'black', 'draw'], name='winner')
# Your code here
chess_games.opening_name.map( lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()).value_counts()
# Your code here
chess_games.loc[chess_games.winner == 'white', ['white_id', 'victory_status']].victory_status.value_counts()
# Your code here
chess_games.nlargest(20, 'turns')
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
# Your code here
(kepler.koi_pdisposition + ' - ' + kepler.koi_disposition).value_counts()
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
print(wine_reviews.head())
print(ramen_reviews.head())
# Your code here
wine_reviews.points = wine_reviews.points.map(lambda p: (p-80)/4 + 1)
wine_reviews
# Your code here
ser = ramen_reviews.loc[ramen_reviews.Stars != 'Unrated', 'Stars']
s = ser.rename('Ramen Ratings')
s.astype('float64')
s.value_counts()
def r_f(p):
    p = float(p)
    a = p - int(p)
    if a > 0.25 and a < 0.75:
        return int(p) + 0.5
    elif a >= 0.75:
        return int(p) + 1
    else:
        return p - a
# Your code here
ser = ramen_reviews.loc[ramen_reviews.Stars != 'Unrated', :]
s = ser.Stars.apply(r_f)
s.value_counts()