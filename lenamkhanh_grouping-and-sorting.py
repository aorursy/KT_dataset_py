import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
check_q1(pd.DataFrame())
# Your code here
common_wine_reviewers = reviews.groupby("taster_twitter_handle").taster_twitter_handle.count()
print(common_wine_reviewers)
check_q1(common_wine_reviewers)
# Your code here
best_wine = reviews.groupby("price").points.max()
print(best_wine)
check_q2(best_wine)
# Your code here
wine_price_extremes = reviews.groupby("variety").price.agg(['min','max'])
print(wine_price_extremes)
check_q3(wine_price_extremes)
# Your code here
reviewer_mean_ratings = reviews.groupby("taster_name").points.mean()
print(reviewer_mean_ratings)
check_q4(reviewer_mean_ratings)
# Your code here
wine_price_range = reviews.groupby("variety").price.agg(['min','max']).sort_index(by=['min','max'],ascending=False)
print(wine_price_range)
check_q5(wine_price_range)
# Your code here
reviews['n'] = 0
country_variety_pairs = reviews.groupby(['country','variety']).n.count().sort_values(ascending = False)
print(country_variety_pairs)
check_q6(country_variety_pairs)