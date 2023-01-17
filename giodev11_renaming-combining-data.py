import pandas as pd

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from renaming_and_combining import *

pd.set_option('max_rows', 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
reviews.rename(columns = {'region_1' : 'region', 'region_2' : 'locale'})
# Your code here
reviews.rename_axis("wines", axis="rows")
gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"
movie_products.head()
# Your code here
pd.concat([movie_products, gaming_products])
powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")
powerlifting_meets.set_index("MeetID").join(powerlifting_competitors.set_index("MeetID"))
dataset = pd.merge(powerlifting_meets, powerlifting_competitors, how='inner', on = 'MeetID')
dataset
dataset = dataset.rename_axis("MeetID", axis="rows")
dataset = dataset.drop('MeetID', 1)

dataset