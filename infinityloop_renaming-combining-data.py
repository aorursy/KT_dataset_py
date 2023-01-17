import pandas as pd

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from renaming_and_combining import *

pd.set_option('max_rows', 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
answer_1 = reviews.rename(columns={'region_1':'region', 'region_2':'locale'})
print(check_q1(answer_1))
answer_1
# Your code here
reviews.index.name = 'wines'
print(check_q2(reviews))
reviews
gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"
# Your code here
gaming_products
movie_products
print(check_q3(pd.concat([gaming_products, movie_products])))
pd.concat([gaming_products, movie_products])
powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")
# Your code here
powerlifting_meets
powerlifting_competitors
print(check_q4(powerlifting_meets.merge(powerlifting_competitors, on='MeetID').set_index('MeetID')))
powerlifting_meets.merge(powerlifting_competitors, on='MeetID').set_index('MeetID')