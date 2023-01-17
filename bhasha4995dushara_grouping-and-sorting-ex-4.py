import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
check_q1(pd.DataFrame())
reviews.head()
reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()
reviews.groupby('price').points.max().sort_index()
reviews.groupby('variety').price.agg([max,min])
reviews.groupby('taster_name').points.mean()
reviews.groupby('variety').price.agg([min,max]).sort_values(by=['min', 'max'], ascending=False)
#reviews['n'] = 0
#reviews.groupby(['country','variety']).n.count().sort_values(ascending=False)
#or
reviews.groupby(['country', 'variety']).size().sort_values(ascending=False)
