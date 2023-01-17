import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
check_q1(pd.DataFrame())
# Your code here
#print(reviews.groupby('taster_twitter_handle').taster_twitter_handle.count().idxmax())
#print(reviews['taster_twitter_handle'].value_counts().idxmax())
common_wine_reviewers = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()
check_q1(common_wine_reviewers)
# Your code here
best_wine = reviews.sort_values(by='price').groupby('price').points.max()
check_q2(best_wine)
# Your code here
df = reviews.groupby('variety')
wine_price_extremes = pd.DataFrame({'min':df.price.min(), 'max':df.price.max()})
# print(wine_price_extremes)
check_q3(wine_price_extremes)
# Your code here
reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()
check_q4(reviewer_mean_ratings)
# Your code here
df = reviews.groupby('variety')
wine_price_range = pd.DataFrame({'min':df.price.min(), 'max':df.price.max()}).sort_values(by=['min', 'max'], ascending=False)
check_q5(wine_price_range)
# Your code here
reviews['counts'] = 0
country_variety_pairs = reviews.groupby(['country', 'variety']).counts.agg(['count']).sort_values(by='count', ascending=False)
check_q6(country_variety_pairs['count'])