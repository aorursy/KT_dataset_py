import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews
check_q1(pd.DataFrame())
# Your code here
common_wine_reviewers = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()
check_q1(common_wine_reviewers)
# Your code here
best_wine = reviews.groupby('price').points.max().sort_index()
check_q2(best_wine)
# Your code here
wine_price_extremes = reviews.groupby('variety').price.agg([min, max])
check_q3(wine_price_extremes)
# Your code here
reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()
check_q4(reviewer_mean_ratings)
# Your code here
wine_price_range = reviews.groupby('variety').price.agg([min, max]).sort_values(by=['min', 'max'], ascending=False)
check_q5(wine_price_range)
# Your code here
reviews['n'] = 0
country_variety_pairs = reviews.groupby(['country', 'variety']).n.count().sort_values(ascending=False)
check_q6(country_variety_pairs)