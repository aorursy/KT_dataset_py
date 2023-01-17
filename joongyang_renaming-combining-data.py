import pandas as pd

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from renaming_and_combining import *

pd.set_option('max_rows', 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
reviews.rename(columns={'region_1': 'region', 'region_2': 'locale'}, inplace=True)
check_q1(reviews)
# Your code here
# reviews.index.name = "wines"
reviews.rename_axis("wines", axis="rows")
check_q2(reviews)
gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"
# Your code here
print(gaming_products.head())
print(movie_products.head())
powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")
# Your code here
powerlifting_meets.head()
powerlifting_competitors.head()
powerlifting = pd.merge(powerlifting_meets, powerlifting_competitors, how="inner", on="MeetID")
print(powerlifting.head())
powerlifting_2 = powerlifting_meets.set_index("MeetID").join(powerlifting_competitors.set_index("MeetID"))
print(powerlifting_2.head())
check_q4(powerlifting_2)