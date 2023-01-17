import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
#pd.set_option("display.max_rows", 5)

from learntools.core import binder; binder.bind(globals())
from learntools.pandas.grouping_and_sorting import *
print("Setup complete.")
# Your code here
reviews_written = reviews.groupby('taster_twitter_handle').description.count()
print(reviews_written)

q1.check()
#q1.hint()
q1.solution()
best_rating_per_price = reviews.groupby('price').points.max().sort_index()
# print(best_rating_per_price)

q2.check()
q2.hint()
q2.solution()
price_extremes = reviews.groupby('variety').price.agg(['min','max'])
# print(price_extremes)
q3.check()
#q3.hint()
#q3.solution()
sorted_varieties = price_extremes.sort_values(['min','max'], ascending=False)  # descending & ascending 
# print(sorted_varieties)
q4.check()
# q4.hint()
q4.solution()
reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()
print(reviewer_mean_ratings.head(5))
q5.check()
#q5.hint()
#q5.solution()
reviewer_mean_ratings.describe()
country_variety_counts = reviews.groupby(['country','variety']).size().sort_values(ascending = False) # 这里的.size()不能替换为.count(),这两个函数有什么不同？
print(country_variety_counts.head(5))
q6.check()
#q6.hint()
q6.solution()