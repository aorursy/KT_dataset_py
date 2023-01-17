import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
check_q1(pd.DataFrame())
reviews.columns
# Your code here
common_wine_reviewers = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()
common_wine_reviewers
check_q1(common_wine_reviewers)
# Your code here
best_wine = reviews.groupby('price', sort=True).points.max()
best_wine
check_q2(best_wine)
# Your code here
wine_price_extremes = reviews.groupby(['variety']).price.agg([min, max])
wine_price_extremes
check_q3(wine_price_extremes)
#reviews.columns
# Your code here
reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()
reviewer_mean_ratings
check_q4(reviewer_mean_ratings)
# Your code here
wine_price_range = reviews.groupby('variety').price.agg([min, max])\
                    .sort_values(['min', 'max'], ascending=False)
wine_price_range
check_q5(wine_price_range)

# Your code here
country_variety_pairs = reviews.groupby(['country', 'variety']).description.agg([len])
country_variety_pairs = country_variety_pairs.reset_index().sort_values('len', ascending=False)
country_variety_pairs
reviews['n']=0
country_variety_pairs = reviews.groupby(['country', 'variety'])\
                        .n.agg(len).sort_values(ascending=False)
country_variety_pairs
check_q6(country_variety_pairs)