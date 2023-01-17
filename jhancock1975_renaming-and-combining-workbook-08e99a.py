import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.renaming_and_combining import *
print("Setup complete.")
reviews.head()
# Your code here
renamed = reviews.rename(columns={'region_1':'region', 'region_2': 'locale'})

q1.check()
#q1.solution()
reindexed = reviews.copy()
reindexed.index.name='wines'

q2.check()
#q2.solution()
gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"
combined_products = pd.concat((gaming_products, movie_products))

q3.check()
#q3.solution()
powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")
powerlifting_combined = powerlifting_meets.join(powerlifting_competitors, on='MeetID', lsuffix='meets', rsuffix='competitors', how='inner')
print(powerlifting_combined.head())
powerlifting_meets.set_index("MeetID").join(powerlifting_competitors.set_index("MeetID"))
print(powerlifting_combined.head())
q4.check()
q4.solution()