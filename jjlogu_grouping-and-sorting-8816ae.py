import pandas as pd



reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

#pd.set_option("display.max_rows", 5)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.grouping_and_sorting import *

print("Setup complete.")
reviews.head()
# Your code here

reviews_written = reviews.groupby("taster_twitter_handle").taster_twitter_handle.count()



q1.check()

reviews_written
#q1.hint()

#q1.solution()
df = reviews.groupby("price").apply(lambda df: df.loc[df.points.idxmax()])

df.loc[:,['price','points','variety']]
best_rating_per_price = reviews.groupby("price").points.max()



q2.check()

best_rating_per_price
#q2.hint()

q2.solution()
reviews.groupby("price").points.max().sort_index()
price_extremes = reviews.groupby("price").apply(lambda df: df.loc[df.points.idxmax()])



q3.check()
#q3.hint()

#q3.solution()
sorted_varieties = ____



q4.check()
#q4.hint()

#q4.solution()
reviewer_mean_ratings = ____



q5.check()
#q5.hint()

#q5.solution()
reviewer_mean_ratings.describe()
country_variety_counts = ____



q6.check()
#q6.hint()

#q6.solution()