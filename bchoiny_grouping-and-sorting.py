import pandas as pd



reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

pd.set_option("display.max_rows", 10)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.grouping_and_sorting import *

print("Setup complete.")
reviews
reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()


# Your code here

reviews_written = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()



q1.check()
q1.hint()

q1.solution()
reviews.groupby('price').apply(lambda x: x.loc[x.points.idxmax()]).points
best_rating_per_price = reviews.groupby('price').apply(lambda x: x.loc[x.points.idxmax()]).points



q2.check()
q2.hint()

q2.solution()
reviews.groupby('variety').price.agg(['min','max'])
price_extremes = reviews.groupby('variety').price.agg(['min','max'])



q3.check()
q3.hint()

q3.solution()
sorted_varieties = price_extremes.sort_values(by=['min','max'],ascending=False)



q4.check()
q4.hint()

q4.solution()
reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()



q5.check()
q5.hint()

q5.solution()
reviewer_mean_ratings.describe()
reviews.groupby(['country','variety']).variety.count().sort_values(ascending=False)
#country_variety_counts = reviews.groupby(['country','variety']).variety.count().sort_values(ascending=False)

country_variety_counts = reviews.groupby(['country', 'variety']).size().sort_values(ascending=False) #size() is better to use. no need to use a variable

q6.check()
q6.hint()

q6.solution()