import pandas as pd

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 10)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
reviews.groupby(reviews['taster_twitter_handle']).taster_twitter_handle.count()
#common_wine_reviewers = reviews.groupby(reviews.taster_twitter_handle.values).value_counts()
#common_wine_reviewers
#reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()
# Your code here
reviews.groupby('price',sort=True).points.apply(max)
best_wine = reviews.groupby('price',sort=True).points.apply(max)
check_q2(best_wine)
# Your code here
wine_price_extremes = reviews.groupby('variety').price.agg([min, max])
check_q3(wine_price_extremes)
# Your code here
import numpy as np

reviewer_mean_ratings = reviews.groupby('taster_name').points.apply(np.mean)
reviewer_mean_ratings
check_q4(reviewer_mean_ratings)
# Your code here
x = reviews.groupby('variety').price.agg([min, max])
wine_price_range = x.sort_values(by=['min', 'max'], ascending=False)
check_q5(wine_price_range)
# Your code here
x = reviews.loc[:, ['country', 'variety']].groupby('country',).variety.value_counts(ascending=False).sort_values(ascending=False)

country_variety_pairs = x
check_q6(country_variety_pairs)


