import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
check_q1(pd.DataFrame())
common_wine_reviewers = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()
common_wine_reviewers
check_q1(common_wine_reviewers)
best_wine = reviews.groupby('price').points.max()
best_wine
#check_q2(best_wine)
wine_price_extremes = reviews.groupby('variety').price.agg([min, max])
wine_price_extremes
check_q3(wine_price_extremes)
reviewer_mean_ratings = reviews.groupby(['taster_name']).points.mean()
reviewer_mean_ratings
check_q4(reviewer_mean_ratings)
wine_price_range = reviews.groupby(['variety']).price.agg([min,max])
wine_price_range = wine_price_range.sort_values(by=['min', 'max'], ascending=False)
wine_price_range
check_q5(wine_price_range)
reviews['n'] = 0
country_variety_pairs = reviews.groupby(['country', 'variety']).n.count().sort_values(ascending=False)
country_variety_pairs
check_q6(country_variety_pairs)

# I think DataFrameGroupBy.size() may be better (or simpler) than DataFrameGroupBy.[column_name].count() 
# if there's no missing data.

# For example, the result of reviews.groupby('points').points.count() is same with reviews.groupby('points').size().

#And, in the Exercise 6, the answer is:
#reviews['n'] = 0
#reviews.groupby(['country', 'variety']).n.count().sort_values(ascending=False)

#and I think below is better :
#reviews.groupby(['country', 'variety']).size().sort_values(ascending=False)

#note : Dataframe needs to have a "flag", it can be "count" a lot of 0s, or it can be "sum" of a lot of 1s. 
#      Nevertheless, the outcome result is to return the "# of Records".