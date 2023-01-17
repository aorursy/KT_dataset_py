import pandas as pd

from learntools.advanced_pandas.renaming_and_combining import *

pd.set_option('max_rows', 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
reviews = reviews.rename({'region_1': 'region', 'region_2': 'locale'}, axis='columns')
reviews
check_q1(reviews)
reviews = reviews.rename_axis("wines", axis="rows")
reviews
check_q2(reviews)
gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"
products = pd.concat([gaming_products, movie_products])
products
check_q3(products)
powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")
powerlifting_meets.head()
powerlifting_competitors.head()
powerlifting_df = powerlifting_meets.set_index('MeetID').join(powerlifting_competitors.set_index('MeetID'))
powerlifting_df
check_q4(powerlifting_df)