import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
check_q1(pd.DataFrame())

common_wine_reviewers = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()
print(common_wine_reviewers)
check_q1(common_wine_reviewers)

best_wine = reviews.groupby('price').points.max()
print(best_wine)
check_q2(best_wine)

wine_price_extremes = reviews.groupby('variety').price.agg([min, max])
print(wine_price_extremes)
check_q3(wine_price_extremes)

reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()
print(reviewer_mean_ratings)
check_q4(reviewer_mean_ratings)
wine_price_range = reviews.groupby('variety').price.agg([min, max])
wine_price_range = wine_price_range.sort_values(['min', 'max'], ascending=False)
print(wine_price_range)
check_q5(wine_price_range)
reviews['n']=0
country_variety_pairs = reviews.groupby(['country','variety']).new.count().sort_values(ascending=False)
check_q6(country_variety_pairs)