import pandas as pd



reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

#pd.set_option("display.max_rows", 5)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.grouping_and_sorting import *

print("Setup complete.")
# Your code here

reviews_written =reviews.groupby('taster_twitter_handle')['taster_twitter_handle'].count()



q1.check()
#q1.hint()

#q1.solution()
best_rating_per_price = reviews.groupby('price')['points'].max()



q2.check()
#q2.hint()

#q2.solution()
var_max = reviews.groupby('variety')['price'].max()

var_max.name = 'max'

var_min = reviews.groupby('variety')['price'].min()

var_min.name = 'min'

price_extremes = pd.concat([var_min,var_max],axis=1)

q3.check()
#q3.hint()

#q3.solution()
foo = price_extremes.copy()

foo.sort_values(['min','max'],ascending=False)



sorted_varieties =foo.sort_values(['min','max'],ascending=False)



q4.check()
#q4.hint()

#q4.solution()


reviewer_mean_ratings = reviews.groupby('taster_name')['points'].mean()

q5.check()
#q5.hint()

#q5.solution()
reviewer_mean_ratings.describe()
reviews.groupby(['country','variety'])['winery'].count().sort_values(ascending=False)



country_variety_counts = reviews.groupby(['country','variety']).size().sort_values(ascending=False)



q6.check()
#q6.hint()

#q6.solution()