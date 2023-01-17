
import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
chess_games
check_q1(pd.DataFrame())
chess_games.head()
# first implementation
#all_count = len(chess_games.index)
#white_wins = len(chess_games[chess_games.winner == 'white'])
#blach_wins = len(chess_games[chess_games.winner == 'black'])
#draws = len(chess_games[chess_games.winner == 'draw'])
#a1 = pd.Series({'white': white_wins/all_count, 'black': blach_wins/all_count, 'draw': draws/all_count}, name='winner')

#better one - USE VALUE_COUNTS! :D
a1 = chess_games['winner'].value_counts() / len(chess_games)
check_q1(a1)
# first implementation
#a2 = (chess_games
#          .assign(opening_archetype = chess_games.opening_name.map(
#             lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()
#            ))
#      .opening_archetype.value_counts())

#better one
a2 = (chess_games
      .opening_name
      .map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip())
      .value_counts()
)

check_q2(a2)

# first try
#chess_games['n'] = 0
#a3 = chess_games.groupby(['white_id', 'victory_status']).n.count()

#correct answer
a3  = (chess_games
       .assign(n=0)
       .groupby(['white_id', 'victory_status'])
       .n
       .apply(len)
       .reset_index()
)

check_q3(a3)


def print_grouped_df(grouped_df):
    for key, item in grouped_df:
        print(key + ": " + grouped_df.get_group(key), "\n\n")
# Your code here
white_results  = (chess_games
       .assign(n=0)
       .groupby(['white_id', 'victory_status'])
       .n
       .apply(len)
       .reset_index()
       .pipe(lambda df: df.loc[df.white_id.isin(chess_games.white_id.value_counts().head(20).index)])
                 )


check_q4(white_results)
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
# Your code here
a5 = (kepler
      .assign(n=0)
      .groupby(['koi_pdisposition', 'koi_disposition'])
      .n.count()
    
)
a5
check_q5(a5)
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
#print(wine_reviews.head())
#print(ramen_reviews.head())
wine_reviews.head()
wine_reviews.describe()
#from math import *
a6 = (((wine_reviews['points'] - 80)/4)
      .value_counts()
      .sort_index()
      .rename('Wine Ratings')
)
check_q6(a6)

# Your code here
ramen_reviews.head()
a7 = (ramen_reviews.Stars
      .replace('Unrated', None).dropna()
      .astype('float64')
      .value_counts()
      .rename_axis('Ramen Ratings')
      .sort_index()
)
a7
check_q7(a7)

# Your code here
def round_to_05_naive(value):
    decimal_part = value - int(value)
    if decimal_part <= 0.25:
        return value - decimal_part
    elif (decimal_part > 0.25) and (decimal_part <= 0.5):
        return value + (0.5 - decimal_part)
    elif (decimal_part > 0.5) and (decimal_part <= 0.75):
        return value - (decimal_part + 0.5)
    elif (decimal_part > 0.75) and (decimal_part <= 1):
        return value + (1 - decimal_part)

def round_to_05_simple(value):
    return round(value * 2) / 2
    
    
a8 = (ramen_reviews.Stars
      .replace('Unrated', None).dropna()
      .astype('float64')
      .map(round_to_05_simple)
      .value_counts()
      .rename_axis('Ramen Ratings')
      .sort_index()
)
a8
check_q8(a8)