import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
check_q1(pd.DataFrame())
# reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()
reviews['taster_twitter_handle'].value_counts().sort_index()
reviews.groupby('price').points.max().sort_index()
reviews.groupby(['variety'])['price'].agg(['min', 'max'])
check_q4(reviews.groupby(['taster_name'])['points'].mean())
check_q5(reviews.groupby(['variety'])['price'].agg([min, max]).sort_values(by=['min', 'max'], ascending=False))
reviews['count'] = 0
check_q6(reviews.groupby(['country', 'variety'])['count'].count().sort_values(ascending=False))