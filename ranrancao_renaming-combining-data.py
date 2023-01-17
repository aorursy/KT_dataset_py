import pandas as pd

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from renaming_and_combining import *

pd.set_option('max_rows', 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
#print (answer_q1())

reviews.rename(columns = {'region_1':'region', 'region_2':'locale'})
# Your code here
#print (answer_q2())

reviews.rename_axis("wines", axis = "rows")

gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"
gaming_products.head()
movie_products.head()
# Your code here
pd.concat([gaming_products, movie_products])
#print (answer_q3())
powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")
powerlifting_meets.head()
powerlifting_competitors.head()
# Your code here
#print (answer_q4())

powerlifting_meets.set_index("MeetID").join(powerlifting_competitors.set_index("MeetID"))