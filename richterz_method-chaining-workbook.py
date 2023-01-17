import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
# Your code here
chess_games['winner'].value_counts()/len(chess_games)

# Your code here
(chess_games['opening_name']
             .map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip())
             .value_counts())
# Your code here
(chess_games
    .assign(n=0)
    .groupby(['white_id','victory_status'])
    ['n']
    .apply(len)
    .reset_index()
   )

#chess_games.head()
#answer_q3()
# Your code here
(chess_games
    .assign(n=0)
    .groupby(['white_id','victory_status'])
    ['n']
    .apply(len)
    .reset_index()
    .pipe(lambda df: df.loc[df['white_id'].isin(chess_games['white_id'].value_counts().head(20).index)]) #.sort_values(by='n',ascending=False).iloc[:20])
   )


kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
# Your code here
kepler.groupby(by=['koi_pdisposition','koi_disposition']).apply(len)



wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
print(wine_reviews.head())
print(ramen_reviews.head())
# Your code here
wine_reviews.rename(columns={'points':'Wine Ratings'})['Wine Ratings'].map(lambda x: int((x-80)/4))


# Your code here
import numpy as np
ramen_reviews['Stars'].replace('Unrated', np.NaN).dropna().rename("Ramen Reviews").astype('float64').value_counts().sort_index()




# Your code here
import numpy as np
(ramen_reviews['Stars']
            .replace('Unrated', np.NaN)
            .dropna()
            .astype('float64')
            .map(lambda x: round(x * 2)/2)
            .rename("Ramen Reviews")
            .value_counts()
            .sort_index()
)

