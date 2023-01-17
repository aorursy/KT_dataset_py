import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
# Your code here
df_size = len(chess_games)
winner_proportion = (chess_games
                         .winner
                         .value_counts()
                         .apply(lambda x: x/df_size)
                    )
check_q1(winner_proportion)
# Your code here
# this is another way to answer but is not accepted
(chess_games
     .assign(opening_name = chess_games.opening_name.apply(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()))
     .groupby(by=['opening_name'])
     .size()
     .sort_values(ascending=False)
)



# this one is the correct
(chess_games
     .opening_name
     .map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip())
     .value_counts()
)
# The only difference is the name of the column, both solution have same size, order and values, 
# but the accepted answer have no name of the column
# This lastone looks simpler and better
# Your code here
# Doing with rename and count functions
check_q3(chess_games.groupby(['white_id', 'victory_status'])
             .count()
             .id
             .rename('n')
             .reset_index()
        )


# doing with assign and size function
check_q3(chess_games
    .assign(n=0)
    .groupby(['white_id', 'victory_status'])
    .n
    .size()
    .reset_index()
)
# Your code here
check_q4(chess_games
    .assign(n=0)
    .groupby(['white_id', 'victory_status'])
    .n
    .size()
    .reset_index()
    .pipe(lambda df: df.loc[df.white_id.isin(chess_games
                                             .white_id
                                             .value_counts()
                                             .head(20)
                                             .index)
                           ]
         ) 
)

kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
# Your code here
check_q5(kepler.groupby(['koi_pdisposition', 'koi_disposition']).size())
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
wine_reviews.head()
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
ramen_reviews.head()
# Your code here
# here need a bit more explanation
# Counts how many reviews have each score and sort in ascending order
check_q6(wine_reviews
             .points
             .dropna()
             .map(lambda p: (p - 80) / 4)
             .value_counts()
             .rename_axis('Wine Ratings')
             .sort_index()
        )


# Your code here
rated = ramen_reviews.Stars!='Unrated'
check_q7(ramen_reviews
         .loc[rated]
         .Stars
         .astype('float64')
         .value_counts()
         .sort_index()
        )
# Your code here
df = (ramen_reviews
         .loc[rated]
         .Stars
         .astype('float64')
         .map(lambda x: int(x) if x - int(x) < 0.5 else int(x) + 0.5)
         .value_counts()
         .sort_index()
         .rename_axis("Ramen Reviews")
 )

[df[i] for i in df.index]

df2 = (ramen_reviews
     .Stars
     .replace('Unrated', None)
     .dropna()
     .astype('float64')
     .map(lambda v: int(v) if v - int(v) < 0.5 else int(v) + 0.5)
     .value_counts()
     .rename_axis("Ramen Reviews")
     .sort_index()
)
sum([df2[i] for i in df2.index])


# how ever I like the plot
check_q8(df2)