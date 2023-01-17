import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
check_q1(pd.DataFrame())
reviews.head()
reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()
# common_wine_reviewers = _______
# check_q1(common_wine_reviewers)
reviews.groupby('price').points.max().sort_index()
# best_wine = ______
# check_q2(best_wine)
reviews.groupby('variety').price.agg([min, max])
# wine_price_extremes = _____
# check_q3(wine_price_extremes)
reviews.groupby('taster_name').points.mean()
# reviewer_mean_ratings = _____
# check_q4(reviewer_mean_ratings)
reviews.groupby('variety').price.agg([min, max]).sort_values(by = ['min', 'max'], ascending = False)
# wine_price_range = ____
# check_q5(wine_price_range)
reviews['n'] = 0
reviews.groupby(['country', 'variety']).n.count().sort_values(ascending = False)
# country_variety_pairs = _____
# check_q6(country_variety_pairs)