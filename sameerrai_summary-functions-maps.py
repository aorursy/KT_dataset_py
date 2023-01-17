import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(20)
reviews.head()
# Your code here
reviews.points.median ()
# Your code here
reviews.country.unique()
# Your code here
countries_max = reviews.country.value_counts()
print (countries_max)
# Your code here
price_median = reviews.price.median ()
reviews.price.map (lambda p: p - price_median)
# Your code here
reviews['points_to_price'] = reviews.points/reviews.price
max_index = reviews.points_to_price.idxmax (skipna = True)
reviews.loc [max_index, 'title']

# Your code here
count_tropical = reviews['description'].str.contains('tropical').value_counts ()[True]
count_fruity = reviews['description'].str.contains('fruity').value_counts ()[True]
# print(count_tropical)
# print(count_fruity)
pd.Series([count_tropical, count_fruity], index=['Tropical','Fruity'])
# Your code here

reviews_dropped_rows = reviews.dropna (subset = ['country','variety'])
# print ("Before dropping empty values, there were %d rows and %d columns \n" % (reviews.shape[0],  reviews.shape[1]))
# print ("Before dropping empty values, there were %d rows and %d columns \n" % (reviews_dropped_rows.shape[0],  reviews_dropped_rows.shape[1]))

country_winevariety = reviews_dropped_rows ['country'] + ' - ' + reviews_dropped_rows ['variety']
country_wine_count = country_winevariety.value_counts()
print (country_wine_count)
