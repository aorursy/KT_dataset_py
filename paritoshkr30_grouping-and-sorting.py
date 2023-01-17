import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
#pd.set_option("display.max_rows", 5)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.grouping_and_sorting import *
print("Setup complete.")
common_wine_reviewers = reviews.groupby('taster_twitter_handle').size()
q1.solution()
best_wine = reviews.groupby('price').points.max()
best_wine

#q2.solution()
wine_price_extremes = reviews.groupby('variety').price.agg([min, max])
wine_price_extremes
#q3.wine_price_extremes()
wine_price_range = reviews.groupby('variety').price.agg([min, max]).sort_values(['min', 'max'], ascending=False)
wine_price_range

#q4.solution()
reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()
reviewer_mean_ratings
#q5.solution()
reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()
reviewer_mean_ratings
reviews['n'] = 0
country_variety_pairs = reviews.groupby(['country', 'variety']).n.count().sort_values(ascending=False)
country_variety_pairs
#q6.solution()