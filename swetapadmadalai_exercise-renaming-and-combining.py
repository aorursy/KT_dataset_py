import pandas as pd



reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.renaming_and_combining import *

print("Setup complete.")
reviews.head()
# Your code here

import pandas as pd

renamed = reviews.rename(columns={"region_1": "region", "region_2":"locale"})

renamed

# Check your answer

#q1.check()
#q1.hint()

#q1.solution()
reindexed = reviews.rename_axis("wines", axis='columns')

reindexed

# Check your answer

#q2.check()
#q2.hint()

#q2.solution()
gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")

gaming_products['subreddit'] = "r/gaming"

movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")

movie_products['subreddit'] = "r/movies"
movie_products['subreddit']
gaming_products['subreddit']
movie_products

gaming_products
combined_products = pd.concat([gaming_products,movie_products])

combined_products

# Check your answer

#q3.check()
#q3.hint()

#q3.solution()
powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")

powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")
powerlifting_competitors
powerlifting_meets

powerlifting_combined = pd.concat([powerlifting_meets,powerlifting_competitors])

powerlifting_combined

# Check your answer

#q4.check()
#q4.hint()

#q4.solution()