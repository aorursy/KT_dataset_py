import pandas as pd

from learntools.advanced_pandas.renaming_and_combining import *

pd.set_option('max_rows', 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
ans = reviews.rename(columns={'region_1': 'region', 'region_2':'locale'})
check_q1(ans)
ans = reviews.rename_axis("wines", axis="rows")
print(ans.head())
check_q2(ans)
gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"
ans = pd.concat([gaming_products, movie_products])
print(ans.head())
check_q3(ans)
powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")
powerlifting_meets = powerlifting_meets.set_index("MeetID")
powerlifting_competitors = powerlifting_competitors.set_index("MeetID")

ans = powerlifting_meets.join(powerlifting_competitors)

print(ans.head())

check_q4(ans)
