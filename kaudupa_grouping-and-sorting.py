%config IPCompleter.greedy=True
import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head(5)
# reviews.groupby(by='taster_twitter_handle').taster_twitter_handle.count()
common_wine_reviewers = reviews.groupby(by='taster_twitter_handle').taster_twitter_handle.count()
common_wine_reviewers.sort_values(ascending=False).index[0]
# # check_q1(common_wine_reviewers)
reviews.fillna(method='bfill',inplace=True)
best_wine=reviews.sort_values(['price'])
best_wine.set_index('price',inplace=True)
best_wine.points
# Your code here
# best_wine = ______
# check_q2(best_wine)
# wine_price_extremes=reviews.loc[:,['variety','price']]
# # wine_price_extremes.set_index('variety',inplace=True)
# wine_price_extremes.groupby('variety').price.agg([min,max])
reviews.groupby('variety').price.agg([min,max])
# reviews.groupby(['country']).price.agg([len, min, max])
# wine_price_extremes
# Your code here
# wine_price_extremes = _____
# check_q3(wine_price_extremes)
# reviewer_mean_ratings=reviews.loc[:,['taster_name','points']]
# reviewer_mean_ratings.groupby('taster_name').points.agg(['mean'])
#
# Your code here
# reviewer_mean_ratings = _____
# check_q4(reviewer_mean_ratings)
# in below cell what i am doing is first i am going to group the dataframe based on taster_name 
# each taste_name will become individual dataframe and from each df i am going to get average points**
reviews.groupby('taster_name').points.agg(['mean'])
wine_price_range =reviews.loc[:,['variety','price']]
wine_price_range.sort_values('price',inplace=True)
wine_price_range.groupby('variety').price.agg(['min','max'])
# Your code here
# wine_price_range = ____
# check_q5(wine_price_range)
# country_variety_pairs =reviews.loc[:,['country','variety']]
# g=country_variety_pairs.groupby(['country','variety']).country.agg(['count'])
# g.sort_values(by='count').head(5)
# reviews.groupby(['country','variety']).country.count()
reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])
# g=country_variety_pairs.groupby(['country','variety'])
# for con, var in g:
#     print(con)
# Your code here
# country_variety_pairs = _____
# check_q6(country_variety_pairs)