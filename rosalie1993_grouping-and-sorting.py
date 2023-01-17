import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
check_q1(pd.DataFrame())
d = reviews.price.apply(lambda x: x - reviews.price.median())
d
# Your code here
#pd.Series(reviews.taster_twitter_handle.value_counts()) #index=taster_twitter_handle)

common_wine_reviewers = pd.Series(reviews.taster_twitter_handle.value_counts())
common_wine_reviewers
#reviews.taster_twitter_handle.unique()
#check_q1(common_wine_reviewers)
reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()
common_wine_reviewers
#print(answer_q1())
# Your code here#
a = reviews.groupby('price').points.max()
#reviews.groupby('price').points.max().sort_index()
#best_wine = reviews.groupby('price').points.max().sort_values()
#check_q2(reviews.groupby('price').points.max().sort_index
check_q2(a)
# Your code here
reviews.groupby('variety').price.agg([min,max])







reviews.groupby('variety').price.agg([min,max])
wine_price_extremes = reviews.groupby('variety').price.agg([min,max])
check_q3(wine_price_extremes)
reviews.groupby('variety').price.agg([min,max])
# Your code here
#reviews.groupby(taster_name).points.average().mean




reviews.groupby('taster_name').points.mean()
reviewer_mean_ratings =reviews.groupby('taster_name').points.mean()
check_q4(reviewer_mean_ratings)
reviews.groupby('taster_name').points.mean
# Your code here
reviews.groupby('variety').price.agg([min,max]).sort_values(by=['min','max'],ascending=False)




reviews.groupby('variety').price.agg([min,max]).sort_values(by=['min','max'], ascending = False)
wine_price_range =reviews.groupby('variety').price.agg([min,max]).sort_values(by=['min','max'], ascending = False)
check_q5(wine_price_range)
# Your code here
reviews['n'] = 0
reviews.groupby(['country','variety']).n.count()



reviews['n'] = 0
reviews.groupby(['country','variety']).n.value_counts()
country_variety_pairs = reviews.groupby(['country','variety']).n.count().sort_values(ascending = False)
#reviews.groupby(['country','variety']).variety.value_counts()
check_q6(country_variety_pairs)
print(answer_q6())
country_variety_pairs
reviews['n'] = 0
reviews.groupby(['country','variety']).n.count()
