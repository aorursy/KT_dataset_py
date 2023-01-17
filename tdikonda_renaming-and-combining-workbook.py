import pandas as pd

from learntools.advanced_pandas.renaming_and_combining import *

pd.set_option('max_rows', 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
df = reviews.rename(columns={'region_1' : 'region', 'region_2' : 'locale'})
check_q1(df)
df = reviews.rename_axis('wines', axis='rows')
check_q2(df)
gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"
df = pd.concat([gaming_products, movie_products])
check_q3(df)
powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")
#powerlifting_competitors.head()
#powerlifting_meets.head()
df1 = powerlifting_competitors.set_index('MeetID')
df2 = powerlifting_meets.set_index('MeetID')
df3 = df2.join(df1)
check_q4(df3)