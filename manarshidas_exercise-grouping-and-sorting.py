import pandas as pd



reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

#pd.set_option("display.max_rows", 5)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.grouping_and_sorting import *

print("Setup complete.")
reviews_written = reviews.groupby('taster_twitter_handle').size()



reviews_written
best_rating_per_price = reviews.groupby('price')['points'].max().sort_index()



best_rating_per_price
price_extremes = reviews.groupby('variety').price.agg([min, max])

price_extremes
sorted_varieties = price_extremes.sort_values(by=['min', 'max'], ascending=False)

sorted_varieties
reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()

reviewer_mean_ratings
reviewer_mean_ratings.describe()
country_variety_counts = reviews.groupby(['country', 'variety']).size().sort_values(ascending=False)

country_variety_counts