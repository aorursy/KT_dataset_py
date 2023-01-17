import pandas as pd


from learntools.advanced_pandas.renaming_and_combining import *

pd.set_option('max_rows', 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
check_q1(reviews.rename(columns = {'region_1': 'region', 'region_2': 'locale'}))
#reviews.set_index({'index' : "wines"}) # This actually takes an existing column and sets it as index. not advisable
check_q2(reviews.rename_axis("wines", axis='rows'))
gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"
# Create df of products mentioned on either subredit
df = pd.concat([gaming_products, movie_products])
df.head()
df.sort_values(['total_mentions'], ascending = False).name[0:15]

# check_q3(df)
powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")

# Let's take a peak
powerlifting_meets.head()
#powerlifting_competitors.head()
# Your code here
left = powerlifting_meets.set_index(['MeetID'])
right = powerlifting_competitors.set_index(['MeetID'])

df = left.join(right, lsuffix='_meet', rsuffix='_competitor')
df.loc[0]
check_q4(df)