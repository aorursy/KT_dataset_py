import pandas as pd



reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

#pd.set_option("display.max_rows", 5)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.grouping_and_sorting import *

print("Setup complete.")
# Your code here

reviews_written = reviews.groupby('taster_twitter_handle').size()



q1.check()
#q1.hint()

#q1.solution()


import numpy as np

import math

point_list =[]

for price in set(reviews['price'].dropna()):

    point = max(reviews[reviews['price'] == price]['points'])

    point_list.append(point)



    

best_rating_per_price = pd.Series(point_list,index=set(reviews['price'].dropna()))



q2.check()
#q2.hint()

#q2.solution()
price_extremes = pd.DataFrame({'min':reviews.groupby('variety')['price'].min(),

                   'max':reviews.groupby('variety')['price'].max()},

                    index=reviews.groupby('variety').groups.keys())



q3.check()
#q3.hint()

#q3.solution()
sorted_varieties = price_extremes.sort_values(['min','max'],ascending=[False,False])



q4.check()
#q4.hint()

#q4.solution()
reviewer_mean_ratings = reviews.groupby('taster_name')['points'].mean()



q5.check()
#q5.hint()

#q5.solution()
reviewer_mean_ratings.describe()
country_variety_counts = reviews.groupby(['country','variety']).size().sort_values(ascending=False)



q6.check()
#q6.hint()

#q6.solution()