import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
# common_wine_reviewers = _______
# check_q1(common_wine_reviewers)
check_q1(reviews.groupby('taster_twitter_handle').count()['taster_name'])
# Your code here
# best_wine = ______
# check_q2(best_wine)
reviews.sort_values(by="price")
check_q2(reviews.groupby('price').max()['points'])
# Your code here
# wine_price_extremes = _____
# check_q3(wine_price_extremes)
minSeries = reviews.groupby('variety').min()['price']
maxSeries = reviews.groupby('variety').max()['price']
check_q3(pd.DataFrame(data={'min':minSeries,'max': maxSeries}, index=minSeries.index))
# Your code here
# reviewer_mean_ratings = _____
# check_q4(reviewer_mean_ratings)
check_q4(reviews.groupby('taster_name').mean()['points'])
# Your code here
# wine_price_range = ____
# check_q5(wine_price_range)
minSeries =reviews.groupby('variety').min()['price']
maxSeries = reviews.groupby('variety').max()['price']
myData = pd.DataFrame(data={'min': minSeries, 'max': maxSeries}, index=minSeries.index)
check_q5(myData.sort_values(by=['min','max'],ascending=False))
# Your code here
# country_variety_pairs = _____
# check_q6(country_variety_pairs)
reviews['n'] = 0
check_q6(reviews.groupby(['country', 'variety']).count()['n'].sort_values(ascending=False))

