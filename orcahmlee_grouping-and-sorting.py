import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 10)
check_q1(pd.DataFrame())
reviews.head()
#common_wine_reviewers = reviews.taster_twitter_handle.value_counts() # Why is this wrong???
common_wine_reviewers = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()

check_q1(common_wine_reviewers)
''' Answer:
reviews.groupby('price').points.max().sort_index()
'''
best_wine = reviews[['price', 'points']].groupby(by=['price']).max()
best_wine = pd.Series(data=best_wine.points, index=best_wine.index)
check_q2(best_wine)
''' Answer:
reviews.groupby('variety').price.agg([min, max])
'''
max_price = reviews.groupby('variety')['price'].max()
min_price = reviews.groupby('variety').price.min()
wine_price_extremes = pd.DataFrame({'min':min_price, 'max':max_price}, index=max_price.index)
wine_price_extremes = wine_price_extremes[['min', 'max']]

check_q3(wine_price_extremes)
df = pd.DataFrame([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]
                    ],
                   columns=['A', 'B', 'C'])
print(df)
df.agg(['sum', 'min'])
# Your code here
# reviewer_mean_ratings = _____
# check_q4(reviewer_mean_ratings)
df.agg([min, max, sum, 'mean'])
wine_price_range = reviews.groupby('variety').price.agg([min, max]).sort_values(by=['min', 'max'], ascending=False)
check_q5(wine_price_range)
print(answer_q6())
reviews['n'] = 0
country_variety_pairs = reviews.groupby(['country', 'variety']).n.count().sort_values(ascending=False)
check_q6(country_variety_pairs)