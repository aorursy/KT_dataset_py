import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
check_q1(pd.DataFrame())
# Your code here
common_wine_reviewers = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()
print(type(common_wine_reviewers))
print(common_wine_reviewers)
#check_q1(common_wine_reviewers)

my_ans1 = reviews.taster_twitter_handle.value_counts()
print(type(my_ans1))
print(my_ans1)
# Your code here
# index = price
# value = points.max()
# Then, sort the values.
best_wine = reviews.groupby('price').points.max().sort_index()
check_q2(best_wine)
# Your code here
# If you want to get only one output transformation, just use "<xxx>.<func>".
# If you want to get multiple outputs, try to use "<xxx>.agg(<list_of_all_func>)".
wine_price_extremes = reviews.groupby('variety').price.agg([min, max])
check_q3(wine_price_extremes)
# Your code here
reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()
check_q4(reviewer_mean_ratings)

# index = reviewers
# value = average review score
# Your code here
wine_price_range = reviews.groupby('variety').price.agg([min, max]).sort_index(by=['min', 'max'], ascending=False)
check_q5(wine_price_range)
#reviews.groupby('variety').price.agg([min, max]).sort_values(by=['min', 'max'], ascending=False)
#answer_q5()
# Your code here
# country_variety_pairs = _____
# check_q6(country_variety_pairs)
reviews['n'] = 0
country_variety_pairs = reviews.groupby(['country', 'variety']).n.count().sort_values(ascending=False)
check_q6(country_variety_pairs)