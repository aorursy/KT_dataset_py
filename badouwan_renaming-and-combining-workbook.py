import pandas as pd

from learntools.advanced_pandas.renaming_and_combining import *

pd.set_option('max_rows', 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
q1 = reviews.rename(columns={'region_1': 'region', 'region_2': 'locale'})
check_q1(q1)
q2 = reviews
q2.index.name = 'wines'
check_q2(q2)
gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"
q3 = pd.concat([gaming_products, movie_products])
check_q3(q3)
powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")
q4 = powerlifting_meets.set_index('MeetID').merge(powerlifting_competitors.set_index('MeetID'), on='MeetID')
check_q4(q4)