import pandas as pd



reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

#pd.set_option("display.max_rows", 5)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.grouping_and_sorting import *

print("Setup complete.")
reviews.head()
# Your code here

number_reviews = reviews.groupby('taster_twitter_handle')['taster_twitter_handle'].count()

reviews_written = pd.Series(number_reviews, index=number_reviews.index)



q1.check()
#q1.hint()

#q1.solution()
reviews.head()
pricing = reviews.groupby('price')['points'].max()

best_rating_per_price = pd.Series(pricing, index=pricing.index)



q2.check()
#q2.hint()

#q2.solution()
# Group by 'variety'

# use 'agg' method for 'min' and 'max'

variety_group = reviews.groupby('variety')['price'].agg([min,max])

price_extremes = pd.DataFrame(variety_group,index=variety_group.index)



q3.check()
#q3.hint()

#q3.solution()
sorted_varieties = price_extremes.sort_values(by=['min','max'], ascending=False)



q4.check()
q4.hint()

q4.solution()
reviewers = reviews.groupby('taster_name')['points'].mean()

reviewer_mean_ratings = pd.Series(reviewers,index=reviewers.index)



q5.check()
#q5.hint()

#q5.solution()
reviewer_mean_ratings.describe()
reviews.head()
common = reviews.groupby(['country','variety'])['variety'].size().sort_values(ascending=False)

# common = common.reset_index()

common

#country_variety_counts = pd.Series(common, index=common.index)

#country_variety_counts.sort
common = reviews.groupby(['country','variety'])['variety'].size().sort_values(ascending=False)

country_variety_counts = pd.Series(common, index=common.index)



q6.check()
q6.hint()

#q6.solution()