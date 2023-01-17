import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()

x = reviews.groupby('price').points.max().sort_index()
check_q2(x)
reviews.groupby('variety').price.agg['max','min']
x = reviews.groupby('taster_name').points.mean()
check_q4(x)
answer_q4()
reviews.groupby('variety').price.agg(['min','max']).sort_values(by=['min', 'max'], ascending=False)
reviews.groupby(['country', 'variety']).n.count().sort_values(ascending=False)