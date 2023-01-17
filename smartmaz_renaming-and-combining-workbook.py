import pandas as pd

from learntools.advanced_pandas.renaming_and_combining import *

pd.set_option('max_rows', 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
d1=reviews.rename(columns={'region_1':'region', 'region_2':'locale'})
check_q1(d1)
# Your code here
d2=reviews.rename_axis('wines',axis='rows')
check_q2(d2)

gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"
# Your code here
d3=pd.concat([gaming_products, movie_products])
check_q3(d3)
powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")
# Your code here
meets=powerlifting_meets.set_index(['MeetID'])
compts=powerlifting_competitors.set_index(['MeetID'])
d4=meets.join(compts, lsuffix='_MEETS', rsuffix='_COMPTS')
check_q4(d4)