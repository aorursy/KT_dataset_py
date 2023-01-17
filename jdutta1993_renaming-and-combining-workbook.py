import pandas as pd

from learntools.advanced_pandas.renaming_and_combining import *

pd.set_option('max_rows', 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
check_q1(reviews.rename(columns = {'region_1': 'region', 'region_2':'locale'}))
reviews.index.name = 'wines'
check_q2(reviews)
gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"
data_1 = pd.DataFrame(gaming_products)
data_2 = pd.DataFrame(movie_products)
data = pd.concat([data_1, data_2])
print(data)
check_q3(data)
powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")
data = powerlifting_meets.set_index("MeetID").join(powerlifting_competitors.set_index("MeetID"))
check_q4(data)
