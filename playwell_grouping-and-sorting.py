import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
check_q()

# Your code here
reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()

common_wine_reviewers = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()
check_q1(common_wine_reviewers)
# Your code here
reviews.groupby('price').points.max()
best_wine = reviews.groupby('price').points.max()
check_q2(best_wine)
# Your code here
reviews.groupby('variety').price.agg([min,max])
wine_price_extremes = reviews.groupby('variety').price.agg([min,max])
check_q3(wine_price_extremes)
# Your code here
reviews.groupby('taster_name').points.mean()
reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()
check_q4(reviewer_mean_ratings)
# Your code here
va = reviews.groupby('variety').price.agg([min,max])
va.sort_values(by=['min','max'],ascending=False)
wine_price_range =va.sort_values(by=['min','max'],ascending=False)
check_q5(wine_price_range)
# Your code here
reviews['n']=0
gr = reviews.groupby(['country','variety']).n.count()

gr.sort_values(ascending=False)
country_variety_pairs =gr.sort_values(ascending=False)
check_q6(country_variety_pairs)