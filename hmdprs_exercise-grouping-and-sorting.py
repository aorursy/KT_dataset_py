from learntools.core import binder; binder.bind(globals())

from learntools.pandas.grouping_and_sorting import *

print("Setup complete.")



import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

# pd.set_option("display.max_rows", 5)
# fun fact

reviews.loc[reviews[(reviews['points'] >= 80)]['price'].idxmax(), ['title', 'price']]
reviews_written = reviews.groupby('taster_twitter_handle').size()



# check your answer

q1.check()
# q1.hint()

# q1.solution()
best_rating_per_price = reviews.groupby('price')['points'].max().sort_index(ascending=True)



# check your answer

q2.check()
# q2.hint()

# q2.solution()
price_extremes = reviews.groupby('variety')['price'].agg([min, max])



# check your answer

q3.check()
# q3.hint()

# q3.solution()
sorted_varieties = price_extremes.sort_values(by=['min', 'max'], ascending=False)



# check your answer

q4.check()
# q4.hint()

# q4.solution()
# reviewer_mean_ratings = reviews.groupby('taster_name')['points'].mean()

# or

reviewer_mean_ratings = reviews.groupby('taster_name')['points'].agg('mean')



# check your answer

q5.check()
# q5.hint()

# q5.solution()
reviewer_mean_ratings.describe()
country_variety_counts = reviews.groupby(['country', 'variety'])['description'].count().sort_values(ascending=False)



# check your answer

q6.check()
# q6.hint()

# q6.solution()