import pandas as pd

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
check_q1(pd.DataFrame())
# Your code here
common_wine_reviewers = reviews.taster_twitter_handle.value_counts()
# check_q1(common_wine_reviewers)
# answer_q1()
expected = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()

print(expected.sort_index(inplace=True) == common_wine_reviewers.sort_index(inplace=True))
# Your code here
best_wine = reviews.groupby('price').points.max().sort_index()
check_q2(best_wine)
# print(reviews[['points', 'price']].set_index('price').sort_index())

# Your code here
wine_price_extremes = reviews.groupby('variety').price.agg([min, max])
check_q3(wine_price_extremes)
# reviews.groupby('variety').agg({'price': [min, max]}).reset_index()
# Your code here
reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()
reviewer_mean_ratings
check_q4(reviewer_mean_ratings)
# Your code here
wine_price_range = reviews.groupby('variety').price.agg([min, max]).sort_values(['min', 'max'], ascending=[False, False])
check_q5(wine_price_range)
wine_price_range
# Your code here
country_variety_pairs = reviews.groupby(['country', 'variety']).size().sort_values(ascending=False)
check_q6(country_variety_pairs)