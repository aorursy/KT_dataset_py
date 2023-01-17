import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
# Your code here
total = len(chess_games)
one = chess_games.winner.value_counts().map(lambda win: win / total)

check_q1(one)
# Your code here

two = chess_games.opening_name.map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip())
two = two.value_counts()

check_q2(two)
# Your code here

three = chess_games.groupby(['white_id', 'victory_status']).victory_status.count()
three = three.rename('n').reset_index()

check_q3(three)

# Your code here
#get names of 20 most active players - .index returns names as list
four = three.pipe(lambda df: df.loc[df.white_id.isin(chess_games.white_id
                                                    .value_counts()
                                                   .head(20).index)])

check_q4(four)
kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
# Your code here
five = kepler.groupby(['koi_pdisposition', 'koi_disposition']).koi_disposition.count()

check_q5(five)
wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
print(wine_reviews.head())
print(ramen_reviews.head())
# Your code here
six = (wine_reviews.points.dropna().map(lambda p: (p - 80) / 4)
       .value_counts()
       .rename_axis('Wine Ratings').sort_index())



check_q6(six)
# Your code here

#ramen_reviews.groupby(['Brand', 'Variety']).Variety.count().head()
seven = (ramen_reviews.loc[ramen_reviews.Stars != 'Unrated']
        .Stars.astype('float64').value_counts()
        .rename_axis('Ramen Ratings')
        .sort_index())

check_q7(seven)
#ramen_reviews.head(25)
# Your code here

#This one is missing some counts for some reason
eight = (ramen_reviews.loc[ramen_reviews.Stars != 'Unrated']
         .Stars.astype('float64').map(lambda x: round(x * 2) / 2).value_counts()
         .rename_axis('Ramen Reviews')
         .sort_index())

eight2 = (ramen_reviews
         .Stars
         .replace('Unrated', None)
         .dropna()
         .astype('float64')
         .map(lambda v: round(v * 2) / 2)
         .value_counts()
         .rename_axis("Ramen Reviews")
         .sort_index()
             )

#print(eight)
#print(eight2)

check_q8(eight2)