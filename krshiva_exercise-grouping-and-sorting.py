import pandas as pd



reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

#pd.set_option("display.max_rows", 5)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.grouping_and_sorting import *

print("Setup complete.")
# Your code here

reviews_written = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()



q1.check()
#q1.hint()

q1.solution()
best_rating_per_price = reviews_written = reviews.groupby('points')['price']

best_rating_per_price.sort_index(ascending=True)

q2.check()
q2.hint()

q2.solution()
price_max = reviews.groupby('variety')['price'].max()

price_min = reviews.groupby('variety')['price'].min()

price_extremes = pd.DataFrame({'max': price_max, 'min': price_min} )

q3.check()
#q3.hint()

q3.solution()
sorted_varieties = reviews.groupby('variety')['price']



q4.check()
#q4.hint()

q4.solution()
reviewer_mean_ratings = ____



q5.check()
#q5.hint()

#q5.solution()
reviewer_mean_ratings.describe()
country_variety_counts = ____



q6.check()
#q6.hint()

#q6.solution()