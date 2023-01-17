import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
check_q1(pd.DataFrame())
reviews.head()
s = reviews.groupby('taster_twitter_handle')['taster_name'].count()
common_wine_reviewers = s
check_q1(common_wine_reviewers)
reviews.head()
reviews.groupby('price').points.agg(max) #want the max points of a paticular price hence using aggregate function.
best_wine = reviews.groupby('price').points.agg(max)
check_q2(best_wine)
df = pd.DataFrame(index=reviews['variety'])
df['min'] = reviews.groupby('variety').price.agg(min)
df['max'] = reviews.groupby('variety').price.agg(max)
df
#wine_price_extremes = df
#check_q3(wine_price_extremes)
data = reviews.groupby('taster_name').points.mean()
s = pd.Series(data=data)
s
reviewer_mean_ratings = s
check_q4(reviewer_mean_ratings)
df.reset_index()
df
#wine_price_range = df
#check_q5(wine_price_range)
reviews['n'] = 0
reviews['n'] = reviews.groupby(['country', 'variety'])
# country_variety_pairs = _____
# check_q6(country_variety_pairs)