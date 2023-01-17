import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
# Your code here
chess_games.winner.value_counts() / len(chess_games)
check_q1(chess_games.winner.value_counts() / len(chess_games))
answer_q1()
# Your code here
(chess_games
    .opening_name
    .map( lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip())
    .value_counts())
check_q2((chess_games
    .opening_name
    .map( lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip())
    .value_counts()))
# Your code here
chess_games['n']=0
chess_games.groupby(['white_id', 'victory_status']).n.agg([len]).sort_values(by='len')
(chess_games
    .assign(n=0) #Assign new columns to a DataFrame, returning a new object
    .groupby(['white_id', 'victory_status'])
    .n
    .apply(len) #Apply a function along an axis of the DataFrame
    .reset_index() 
#When we reset the index, the old index is added as a column, and a new sequential index is used
)
# Your code here
(chess_games
    .assign(n=0) 
    .groupby(['white_id', 'victory_status'])
    .n
    .apply(len) 
    .reset_index()
    .pipe(lambda df: df.loc[df.white_id.isin(chess_games.white_id.value_counts().head(20).index)])
)
#answer_q4()
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
# Your code here
print(kepler.koi_pdisposition.unique()) #pre disposition
print(kepler.koi_disposition.unique()) #the after status 

kepler['n']=0
kepler.groupby(['koi_pdisposition', 'koi_disposition']).n.agg([len]).reset_index()
#use the chaining technique
(kepler
     .assign(count_n=0)
     .groupby(['koi_pdisposition', 'koi_disposition'])
     .count_n
     .apply(len)
     .reset_index()
)
check_q5(kepler.assign(n=0).groupby(['koi_pdisposition', 'koi_disposition']).n.count())
#answer_q5()
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
print(wine_reviews.head())
print(ramen_reviews.head())
# Your code here
#print(wine_reviews.columns)
print(wine_reviews.points.isnull().sum())
print(wine_reviews.points.unique())
#stars 1 to 5  
#100 is 5 means that 
((wine_reviews['points'] / 20) 
     .value_counts()
     .sort_index()
     .rename_axis('wine_ratings')
)
check_q6((((wine_reviews['points'] -80 ) / 4) 
     .value_counts()
     .sort_index()
     .rename_axis('Wine Ratings')
))
#answer_q6()
ramen_reviews.Stars.isnull().sum()
ramen_reviews.Stars.set_axis
#answer_q7()
# Your code here
(ramen_reviews
     .Stars
     .replace('Unrated', None)
     .dropna()
     .astype(float)
     .value_counts()
     .rename_axis('Ramen Ratings')
     .sort_index()
)
check_q7((ramen_reviews
     .Stars
     .replace('Unrated', None)
     .dropna()
     .astype(float)
     .value_counts()
     .rename_axis('Ramen Ratings')
     .sort_index()
))
# Your code here
(ramen_reviews
     .Stars
     .replace('Unrated', None)
     .dropna()
     .astype(float)
     .map(lambda x: round(x*2)/2)
     .value_counts()
     .rename_axis('Ramen Ratings')
     .sort_index()
)
check_q8((ramen_reviews
     .Stars
     .replace('Unrated', None)
     .dropna()
     .astype(float)
     .map(lambda x: round(x*2)/2)
     .value_counts()
     .rename_axis('Ramen Ratings')
     .sort_index()
))