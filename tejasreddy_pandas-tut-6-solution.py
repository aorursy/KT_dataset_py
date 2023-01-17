import pandas as pd

from learntools.advanced_pandas.renaming_and_combining import *

pd.set_option('max_rows', 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
res = reviews.rename(columns={"region_1":"region", "region_2":"locale"})
print(res)
print(check_q1(res))
res = reviews.rename_axis("wines",axis=0)
print(res)
print(check_q2(res))
gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"
res = pd.concat([gaming_products, movie_products])
print(res)
print(check_q3(res))
powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")
left = powerlifting_meets.set_index(["MeetID"])
right = powerlifting_competitors.set_index(["MeetID"])
res = left.join(right, lsuffix="_meets", rsuffix="_competitors")
print(res)
check_q4(res)