import pandas as pd



reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.renaming_and_combining import *

print("Setup complete.")

reviews.head()
reviews.head()
# Your code here

renamed = reviews.rename(columns=dict(region_1='region',region_2='locale'))

print(renamed)

q1.check()
#q1.hint()

q1.solution()
reindexed = reviews.rename_axis('wines',axis='rows')

print(reindexed)

q2.check()
#q2.hint()

q2.solution()
gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")

gaming_products['subreddit'] = "r/gaming"

gaming_products

movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")

movie_products['subreddit'] = "r/movies"

movie_products
combined_products = pd.concat([gaming_products, movie_products])

print(combined_products)

q3.check()
#q3.hint()

q3.solution()
powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")

print(powerlifting_meets.head)
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")

print(powerlifting_competitors.head)
powerlifting_combined = powerlifting_meets.set_index('MeetID').join(powerlifting_competitors.set_index('MeetID'))

print(powerlifting_combined)

q4.check()
#q4.hint()

q4.solution()