import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
check_q1(pd.DataFrame())
reviews.head()

common_wine_reviewers = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()
check_q1(common_wine_reviewers)

best_wine = reviews.groupby('price').points.max()
best_wine
check_q2(best_wine)
wine_price_extremes = reviews.groupby('variety').price.agg([min, max])
wine_price_extremes
check_q3(wine_price_extremes)

# Your code here
reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()
reviewer_mean_ratings
check_q4(reviewer_mean_ratings)

# Your code here
wine_price_range = reviews.groupby('variety').price.agg([min, max])
wine_price_range = wine_price_range.sort_values(by=['min','max'], ascending=False)
wine_price_range
check_q5(wine_price_range)


country_variety_pairs = reviews.assign(n=0).groupby(['country', 'variety']).n.count().sort_values(ascending=False)
check_q6(country_variety_pairs)