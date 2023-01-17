import pandas as pd

from learntools.advanced_pandas.renaming_and_combining import *

pd.set_option('max_rows', 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
#reviews.rename({"region_1":"region", "region_2":"locale"}, axis="columns",inplace=True)
check_q1(reviews.rename({"region_1":"region", "region_2":"locale"}, axis="columns"))
# Your code here
check_q2(reviews.rename_axis("wines", axis="index"))
gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"
# Your code here
#gaming_products
#gaming_products.merge(movie_products)
pd.concat([gaming_products,movie_products])
check_q3(pd.concat([gaming_products,movie_products]))
powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")
powerlifting_competitors
powerlifting_meets.columns
powerlifting_meets
# Your code here
powerlifting_meets.merge(powerlifting_competitors, on="MeetID").drop_duplicates()
#The question does not say anything about changing the index it just says to combine the
#dataframes into 1 dataframe
answer_q4()
powerlifting_meets.set_index("MeetID").join(powerlifting_competitors.set_index("MeetID"))