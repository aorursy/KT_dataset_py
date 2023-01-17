import pandas as pd
from learntools.advanced_pandas.grouping_and_sorting import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

pd.set_option("display.max_rows", 5)
check_q1(pd.DataFrame())
reviewers = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()
reviewers = reviewers.sort_values(ascending=False)
reviewers.head(4).plot.bar()
reviewers.plot.line()
reviews.head()
wines = reviews.groupby('price').points.agg([max]).sort_index(ascending=True)
#wines.head(6).plot.bar()
reviews.groupby('variety').price.agg([min,max])

# Your code here
reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()
check_q4(reviewer_mean_ratings)
# Your code here
wine_price_range = reviews.groupby('variety').price.agg([min,max]).sort_values(by=['min','max'], ascending = False)
check_q5(wine_price_range)

# Your code here
reviews['n'] = 0
country_variety_pairs = reviews.groupby(['country','variety']).n.count().sort_values(ascending=False)
check_q6(country_variety_pairs)