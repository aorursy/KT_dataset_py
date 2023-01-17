import pandas as pd



reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

#pd.set_option("display.max_rows", 5)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.grouping_and_sorting import *

print("Setup complete.")
reviews.head()
# Your code here

reviews_written = reviews.groupby('taster_twitter_handle').size()

q1.check()

reviews_written
#q1.hint()

#q1.solution()
best_rating_per_price = reviews.groupby('price')['points'].max().sort_index()



q2.check()

best_rating_per_price
#q2.hint()

#q2.solution()
price_extremes = reviews.groupby('variety').price.agg([min, max])



q3.check()

price_extremes
#q3.hint()

#q3.solution()
sorted_varieties = price_extremes.sort_values(by=['min', 'max'], ascending=False)





q4.check()

sorted_varieties
#q4.hint()

#q4.solution()
reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()



q5.check()

reviewer_mean_ratings.head()

#q5.hint()

#q5.solution()
reviewer_mean_ratings.describe()
country_variety_counts =  reviews.groupby(['country', 'variety']).size().sort_values(ascending=False)





q6.check()

country_variety_counts.head()
#q6.hint()

#q6.solution()