import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)

reviews.head()
check_q1(pd.DataFrame())
# Your code here
# index = taster_twitter_handle
# values = # reviews/reviewer

common_wine_reviewers = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()

# common_wine_reviewers = _______
check_q1(common_wine_reviewers)
# srs, index = wine prices, values = max points to that price(from review
# Step 1: group by price, then of those values, find the maximum value of points category

best_wine = reviews.groupby('price').points.max()


check_q2(best_wine)
best_wine
# Min and Max for each variety? create df, index = variety
wine_price_extremes = reviews.groupby(by = 'variety').price.agg([min, max])
check_q3(wine_price_extremes)
# create srs, index = reviewers, values = average review by reviewer
reviewer_mean_ratings = reviews.groupby(by = 'taster_name').points.mean()
check_q4(reviewer_mean_ratings)
# index = variety, value = min and max of price
wine_price_range = reviews.groupby('variety').price.agg([min, max]).sort_values(by = ['min', 'max'] , ascending = False)
check_q5(wine_price_range)
# index = multiindex of {country,variety} sort descending # of varieties
reviews['n'] = 0
reviews.n = reviews.groupby(['n']).variety.value_counts()

country_variety_pairs = reviews.groupby(['country', 'variety']).size().sort_values(ascending = False)
check_q6(country_variety_pairs)