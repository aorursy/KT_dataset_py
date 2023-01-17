import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.method_chaining import *

chess_games = pd.read_csv("../input/chess/games.csv")
check_q1(pd.DataFrame())
chess_games.head()
# Your code here
check_q1((chess_games["winner"].value_counts() / chess_games.count()["id"]))

# notes:
# chess_games.count()["id"] could be replaced by len(chess_games)
# check_q1((chess_games["winner"].value_counts() / chess_games.count()["id"]).apply(lambda v: "%.2f" % v)) # if format was needed

# Your code here
check_q2(chess_games["opening_name"].apply(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip()).value_counts())
# Your code here
# chess_games['n'] = 1
# chess_games.groupby(['white_id', 'victory_status'])["n"].apply(len).reset_index()

# chained mode
check_q3(chess_games
            .assign(n=1)
            .groupby(['white_id', 'victory_status'])["n"]
            .apply(len)
            .reset_index() ## WHY?
        )


# Your code here
def top_20(df):
    # select only the entries from the top 20 players
    df = df.loc[df.white_id.isin(chess_games["white_id"].value_counts().head(20).index)]
    return df

check_q4(chess_games
    .assign(n=1)
    .groupby(['white_id', 'victory_status'])["n"]
    .apply(len)
    .reset_index() ## WHY?
    .pipe(top_20)
    
    # alternative:
    # .pipe(lambda df: df.loc[df.white_id.isin(chess_games.white_id.value_counts().head(20).index)]) 
)


kepler = pd.read_csv("../input/kepler-exoplanet-search-results/cumulative.csv")
kepler
# Your code here
check_q5(kepler
     .assign(n=0)
     .groupby(["koi_pdisposition", "koi_disposition"])["n"]
     .count()
)

wine_reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
ramen_reviews = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv", index_col=0)
print(wine_reviews.head())
print(ramen_reviews.head())
# Your code here
def normalize_points(val):
# Normalize considering integers only (wrong)
#     if (val >= 80 + (20/5 * 4)):
#         val = 5
#     elif (val >= 80 + (20/5 * 3)):
#         val = 4
#     elif (val >= 80 + (20/5 * 2)):
#         val = 3
#     elif (val >= 80 + (20/5 * 1)):
#         val = 2
#     elif (val >= 80 + (20/5 * 0)):
#         val = 1

# Normalize with float values
    val = (val - 80) / 4

    return val

check_q6(wine_reviews["points"].dropna().apply(normalize_points).value_counts().sort_index().rename_axis("Wine Ratings"))


# better answer:
check_q6(((wine_reviews['points'].dropna() - 80) / 4)
    .value_counts()
    .sort_index()
    .rename_axis("Wine Ratings")
)
# Your code here
check_q7(ramen_reviews[ramen_reviews["Stars"] != "Unrated"]["Stars"]
         .astype("float64")
         .value_counts()
         .rename_axis("Ramen Ratings")
         .sort_index())

# answer_q7()
# (ramen_reviews
#     .Stars
#     .replace('Unrated', None)
#     .dropna()
#     .astype('float64')
#     .value_counts()
#     .rename_axis("Ramen Reviews")
#     .sort_index())


# Your code here

# Returns False, but I dont understand why
# check_q8(ramen_reviews[ramen_reviews["Stars"] != "Unrated"]["Stars"]
#          .astype("float64")
#          .map(lambda v: round(v * 2) / 2)
#          .value_counts()
#          .rename_axis("Ramen Ratings")
#          .sort_index())

check_q8(ramen_reviews
     .Stars
     .replace('Unrated', None)
     .dropna()
     .astype('float64')
     .map(lambda v: round(v * 2) / 2)
     .value_counts()
     .rename_axis("Ramen Reviews")
     .sort_index()
)