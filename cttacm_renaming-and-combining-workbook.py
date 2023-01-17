import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.renaming_and_combining import *
print("Setup complete.")
reviews.head()
# Your code here
renamed = reviews.rename(columns={'region_1':'region','region_2':'locale'})

q1.check()
#q1.hint()
#q1.solution()
reindexed = reviews.rename_axis("wines")

q2.check()
#q2.hint()
#q2.solution()
gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
print(gaming_products.head(3))
movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"
print(movie_products.head(3))
combined_products = pd.concat([gaming_products,movie_products])
print(combined_products.head(5))
q3.check()
#q3.hint()
q3.solution()
powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
print(powerlifting_meets.head(3))
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")
print(powerlifting_competitors.head(3))
powerlifting_combined = powerlifting_meets.set_index('MeetID').join(powerlifting_competitors.set_index('MeetID'))

q4.check()
#q4.hint()
q4.solution()