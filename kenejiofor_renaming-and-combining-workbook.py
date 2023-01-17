import pandas as pd

from learntools.advanced_pandas.renaming_and_combining import *

pd.set_option('max_rows', 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
reviews.rename(columns={'region_1':'region','region_2':'Appellation'})
# Your code here
#print(answer_q2())
gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"
# Your code here
#gaming_products.head()
#movie_products.head()
#gaming_products.shape
#movie_products.shape
subreddits = pd.concat([gaming_products,movie_products])
subreddits.shape

powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")
# Your code here
#powerlifting_meets.head()
#powerlifting_competitors.head()
powerlifting_data = pd.merge(powerlifting_meets,powerlifting_competitors.rename(columns={'MeetID':'MeetID'}),on='MeetID',how='left')
powerlifting_data.head()