import pandas as pd

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
check_q1(pd.DataFrame())
# Your code here
# common_wine_reviewers = _______
# check_q1(common_wine_reviewers)

common_wine_reviewers=reviews['taster_twitter_handle'].groupby(by = reviews.taster_twitter_handle).count()

check_q1(common_wine_reviewers)
# Your code here
# best_wine = ______
# check_q2(best_wine)

reviews.groupby('price').points.max()
check_q2(reviews.groupby('price').points.max())
# Your code here
# wine_price_extremes = _____
# check_q3(wine_price_extremes)

minimum = pd.Series(data=reviews.groupby('variety').price.min(),name='min')
maximum = pd.Series(data=reviews.groupby('variety').price.max(),name='max')
pd.concat([minimum,maximum],axis = 1)
check_q3(pd.concat([minimum,maximum],axis = 1))
# Your code here
# reviewer_mean_ratings = _____
# check_q4(reviewer_mean_ratings)

reviews.groupby('taster_name').points.mean()
check_q4(reviews.groupby('taster_name').points.mean())
# Your code here
# wine_price_range = ____
# check_q5(wine_price_range)

minimum = pd.Series(data=reviews.groupby('variety').price.min(),name='min')
maximum = pd.Series(data=reviews.groupby('variety').price.max(),name='max')
df = pd.concat([minimum,maximum],axis = 1)
df.sort_values(by=['min','max'],ascending=False)
check_q5(df.sort_values(by=['min','max'],ascending=False))
# Your code here
# country_variety_pairs = _____
# check_q6(country_variety_pairs)
reviews['n']=0
reviews.groupby(['country','variety']).n.count().sort_values(ascending=False)
check_q6(reviews.groupby(['country','variety']).n.count().sort_values(ascending=False))