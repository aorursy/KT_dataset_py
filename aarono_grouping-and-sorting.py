import pandas as pd
import numpy as np
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)

reviews.head()
check_q1(pd.DataFrame())
# Your code here
# common_wine_reviewers = _______
# check_q1(common_wine_reviewers)

common_wine_reviewers = reviews.taster_twitter_handle.value_counts().sort_values(ascending=False)

common_wine_reviewers
#reviews.groupby('taster_twitter_handle').taster_twitter_handle.count().sort_values(ascending=False)

#check_q1(common_wine_reviewers)
# Your code here
# best_wine = ______
# check_q2(best_wine)

best_wine = reviews.groupby('price').points.max()
check_q2(best_wine)

# Your code here
# wine_price_extremes = _____
# check_q3(wine_price_extremes)
reviews.groupby(['variety']).price.agg([min, max])
check_q3(reviews.groupby(['variety']).price.agg([min, max]))
# Your code here
# reviewer_mean_ratings = _____
# check_q4(reviewer_mean_rating)
reviewer_mean_ratings = reviews.groupby(['taster_name']).points.agg(np.mean)

check_q4(reviewer_mean_ratings)
# Your code here
# wine_price_range = ____
# check_q5(wine_price_range)

wine_price_range = reviews.groupby(['variety']).price.agg(['min','max']).sort_values(by=['min','max'], ascending=False)

check_q5(wine_price_range)
# Your code here
# country_variety_pairs = _____
# check_q6(country_variety_pairs)