import pandas as pd



reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.renaming_and_combining import *

print("Setup complete.")
reviews.head()
# Your code here

renamed = reviews.rename(columns={'region_1':'region','region_2':'locale'})



q1.check()
# q1.hint()

# q1.solution()
reindexed = reviews.rename_axis('wines')



q2.check()
# q2.hint()

# q2.solution()
gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")

gaming_products['subreddit'] = "r/gaming"

movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")

movie_products['subreddit'] = "r/movies"
combined_products = pd.concat([gaming_products,movie_products])



q3.check()
# q3.hint()

# q3.solution()

# movie_products.head()

powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")

powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")
powerlifting_meets.shape, powerlifting_competitors.shape
# powerlifting_combined = powerlifting_meets.join(powerlifting_competitors,on='MeetID',how='left',lsuffix='_left',rsuffix='_right')

# print(powerlifting_combined.shape)

# # (8482, 26) inner join all the columns + key columns





powerlifting_combined = pd.merge(powerlifting_meets,powerlifting_competitors,on='MeetID',how='inner',validate='one_to_many')

powerlifting_combined.set_index('MeetID',inplace=True)

# print(powerlifting_combined.shape)

# print(powerlifting_combined.columns)

# # # (386414, 24) inner join all columns except key col + only one key column



q4.check()
# q4.hint()

q4.solution()