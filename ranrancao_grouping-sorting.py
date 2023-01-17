import pandas as pd

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
# common_wine_reviewers = _______
# check_q1(common_wine_reviewers)
#print (answer_q1())
reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()
# Your code here
# best_wine = ______
# check_q2(best_wine)
#print (answer_q2())
reviews.groupby('price').points.max().sort_index()
# Your code here
# wine_price_extremes = _____
#check_q3(wine_price_extremes)
#print (answer_q3())

reviews.groupby('variety').price.agg([min, max])
# Your code here
# reviewer_mean_ratings = _____
# check_q4(reviewer_mean_rating)
#print (answer_q4())
reviews.groupby('taster_name').points.mean()
# Your code here
# wine_price_range = ____
# check_q5(wine_price_range)
#print(answer_q5())

reviews.groupby('variety').price.agg([min, max]).sort_values(by = ['min', 'max'], ascending = 'false')
# Your code here
# country_variety_pairs = _____
# check_q6(country_variety_pairs)
reviews['n'] = 0
reviews.groupby (['country', 'variety']).n.count().sort_values(ascending = False)


#print (answer_q6())