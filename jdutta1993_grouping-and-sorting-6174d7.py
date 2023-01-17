import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
data = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()
print(data)
common_wine_reviewers = data
check_q1(common_wine_reviewers)
data = reviews.groupby('price').points.max()
print(data)
best_wine = data
check_q2(best_wine)
data = reviews.groupby('variety').price.agg(['min', 'max'])
print(data)
wine_price_extremes = data
check_q3(wine_price_extremes)
data = reviews.groupby('taster_name').points.mean()
print(data)
reviewer_mean_ratings = data
check_q4(reviewer_mean_ratings)
data = reviews.groupby('variety').price.agg(['min', 'max']).sort_values(by = ['min', 'max'], ascending = False)
print(data)
wine_price_range = data
check_q5(wine_price_range)
reviews['country_variety'] = 0
data = reviews.groupby(['country', 'variety']).country_variety.count().sort_values(ascending = False)
print(data)
country_variety_pairs = data
check_q6(country_variety_pairs)