import pandas as pd



reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

#pd.set_option("display.max_rows", 5)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.grouping_and_sorting import *

print("Setup complete.")
# Your code here

reviews_written=reviews.groupby('taster_twitter_handle').size()

print(reviews_written)
best_rating_per_price = reviews.groupby('price').points.max()

print(best_rating_per_price)
price_extremes = reviews.groupby('variety').price.agg([min,max])

print(price_extremes)
sorted_varieties = price_extremes.sort_values(by=['min','max'], ascending=False)

print(sorted_varieties)
reviewer_mean_ratings =reviews.groupby('taster_name').points.mean()

print(reviewer_mean_ratings)
reviewer_mean_ratings.describe()
country_variety_counts = reviews.groupby(['country','variety']).variety.count().sort_values(ascending=False)

print(country_variety_counts)