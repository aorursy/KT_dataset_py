import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
check_q1(pd.DataFrame())
q1 = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()
q1
# Your code here
# common_wine_reviewers = _______
check_q1(q1)
q2 = reviews.groupby('price').points.max().sort_index()
q2
# Your code here
# best_wine = ______

reviews.groupby('variety').price.agg([min, max])
# Your code here
# wine_price_extremes = _____

q4 = reviews.groupby('taster_name').points.mean()
q4

# Your code here
# reviewer_mean_ratings = _____
# check_q4(q4)
q5 = reviews.groupby('variety').price.agg([min, max]).sort_values(by=['min', 'max'], ascending=False)
# Your code here
# wine_price_range = ____
check_q5(q5)

reviews['n'] = 0
x = reviews.groupby(['country', 'variety']).n.count().sort_values(ascending=False)

# Your code here
# country_variety_pairs = _____
# answer_q6()
