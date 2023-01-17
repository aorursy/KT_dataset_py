# Loading packages in

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# import wine reviews dataset

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

reviews.head()
cheapest = pd.DataFrame(reviews.groupby('points')['price'].min())



print(cheapest)
# the function "cheapest_wine" returns you the cheapest options for a specific rating

def cheapest_wine(rating=80):

    return reviews.loc[(reviews.price == float(cheapest.loc[rating])) & (reviews.points == rating),['title','price', 'points']]



cheapest_wine(92)
# the only argument you have to give to the function is the rating you are interested in. 

# As a default a rating of "80" is stated.

cheapest_wine()