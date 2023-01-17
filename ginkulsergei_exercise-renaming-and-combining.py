import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.renaming_and_combining import *
print("Setup complete.")
reviews.head()

#reviews.columns = ['country', 'description', 'designation', 'points', 'price', 'province',
  #     'region', 'locale', 'taster_name', 'taster_twitter_handle', 'title',
  #     'variety', 'winery']

reviews.head()
# Your code here
renamed = reviews.rename(columns = {"region_1" : "region", "region_2" : "locale"})

# Check your answer
q1.check()
#q1.hint()
q1.solution()
reviews.rename_axis(index = "wines", axis = "index")
reindexed = reviews.rename_axis(index = "wines", axis = "index")


# Check your answer
q2.check()
#q2.hint()
q2.solution()
gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"
pd.concat([gaming_products, movie_products])
combined_products = pd.concat([gaming_products, movie_products])

# Check your answer
q3.check()
#q3.hint()
q3.solution()
powerlifting_meets

powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")
pd.merge(powerlifting_meets, powerlifting_competitors, on = "MeetID")

powerlifting_meets.set_index("MeetID").join(powerlifting_competitors.set_index("MeetID"), on = 'MeetID')
powerlifting_combined = powerlifting_meets.set_index("MeetID").join(powerlifting_competitors.set_index("MeetID"), on = 'MeetID')

# Check your answer
q4.check()
q4.hint()
q4.solution()