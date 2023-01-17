import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
check_q1(pd.DataFrame())
reviews.columns
vc = reviews.taster_twitter_handle.value_counts()
vc
type(vc)
vc.index[0]
# Your code here
a1 = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()
common_wine_reviewers = a1
check_q1(common_wine_reviewers)
reviews.price.count()
reviews.groupby('price').price.count()
reviews.groupby('price').points.max()
# Your code here
best_wine = reviews.groupby('price').points.max()
check_q2(best_wine)
# Your code here
wine_price_extremes = reviews.groupby('variety').price.agg([min,max])
check_q3(wine_price_extremes)
wine_price_extremes
# Your code here
reviewer_mean_rating = reviews.groupby('taster_name').points.mean()
check_q4(reviewer_mean_rating)
# Your code here
wine_price_range = reviews.groupby('variety').price.agg([min,max]).sort_values(by = ['min','max'], ascending = False)
wine_price_range
check_q5(wine_price_range)

sum(reviews.description.isnull())
# Your code here
reviews['n'] = 1
country_variety_pairs = reviews.groupby(['country', 'variety']).n.sum().sort_values(ascending = False)
#  = _____
check_q6(country_variety_pairs) 