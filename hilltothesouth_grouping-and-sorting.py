import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option ('max_rows', 5)
check_q1(pd.DataFrame())
common_wine_reviewers = reviews['taster_twitter_handle'].groupby (reviews.taster_twitter_handle).count()
check_q1(common_wine_reviewers)
best_wine = reviews.groupby('price').points.max()
check_q2(best_wine)
wine_price_extremes = reviews.groupby ('variety').price.agg ([min, max])
check_q3(wine_price_extremes)
reviewer_mean_ratings = reviews.groupby ('taster_name').points.mean ()
check_q4(reviewer_mean_ratings)
varieties_first = reviews.groupby ('variety')
range_of_prices = varieties_first ['price'].agg ([min, max])
wine_price_range = range_of_prices.sort_values (by = ['min', 'max'], ascending = False)
check_q5(wine_price_range)
reviews ['n'] = 0
country_variety_pairs = reviews.groupby (['country', 'variety']).n.count ().sort_values (ascending = False)
check_q6(country_variety_pairs)