import pandas as pd

from learntools.advanced_pandas.renaming_and_combining import *

pd.set_option('max_rows', 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
a1 = reviews.rename(columns={'region_1': 'region', 'region_2': 'locale'})
check_q1(a1)
# Your code here
a2 = reviews.rename_axis('wines', axis='rows')
check_q2(a2)
gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"
# Your code here
a3 = pd.concat([gaming_products, movie_products])
check_q3(a3)

powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")

# Your code here
left = powerlifting_meets.set_index('MeetID')
right = powerlifting_competitors.set_index('MeetID')
a4 = left.join(right, lsuffix='_meets', rsuffix='_comps')
check_q4(a4)