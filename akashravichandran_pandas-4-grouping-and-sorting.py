import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
check_q1(pd.DataFrame())
# Your code here
# common_wine_reviewers = _______
# check_q1(common_wine_reviewers)
# reviews.head()
res = reviews.copy()
df = res.groupby('taster_twitter_handle')['taster_twitter_handle'].count()
print(df)
print(check_q1(df))
# Your code here
# best_wine = ______
# check_q2(best_wine)
res = reviews.copy()
df = res.groupby('price')['points'].max().sort_index()
print(df)
print(check_q2(df))
# Your code here
# wine_price_extremes = _____
# check_q3(wine_price_extremes)
res = reviews.copy()
df = res.groupby('variety')['price'].agg([min, max])
print(df)
print(check_q3(df))
# Your code here
# reviewer_mean_ratings = _____
# check_q4(reviewer_mean_ratings)
res = reviews.copy()
df = res.groupby('taster_name')['points'].mean()
print(df)
print(check_q4(df))
# Your code here
# wine_price_range = ____
# check_q5(wine_price_range)
res = reviews.copy()
df = res.groupby('variety')['price'].agg(['min','max']).sort_values(by=['min', 'max'], ascending=False)
print(df)
print(check_q5(df))
# Your code here
# country_variety_pairs = _____
# check_q6(country_variety_pairs)
res = reviews.copy()
res['n'] = 0
df = res.groupby(['country', 'variety']).n.count().sort_values(ascending=False)
print(df)
print(check_q6(df))