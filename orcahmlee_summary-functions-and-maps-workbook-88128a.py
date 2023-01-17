import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
check_q1(reviews.points.median())
check_q2(reviews.country)
check_q3(reviews.country.value_counts())
median_price = reviews.price.median()
data = reviews.price.map(lambda v : v - median_price)
check_q4(data)
''' Answer:
reviews.loc[(reviews.points / reviews.price).argmax()].title
'''
#(reviews.points / reviews.price).max() # get the maximum value
#(reviews.points / reviews.price).idxmax() # get the row_index of the maximum value
#(reviews.points / reviews.price).argmax() # Deprecated since version 0.21.0
data = reviews.loc[(reviews.points / reviews.price).idxmax()].title
check_q5(data)
''' Answer:
tropical_wine = reviews.description.map(lambda r: "tropical" in r).value_counts()
fruity_wine = reviews.description.map(lambda r: "fruity" in r).value_counts()
pd.Series([tropical_wine[True], fruity_wine[True]], index=['tropical', 'fruity'])
'''
tropical = reviews.description.str.contains('tropical').sum()
fruity = reviews.description.str.contains('fruity').sum()
data = pd.Series(data=[tropical, fruity], index=['tropical', 'fruity'])
check_q6(data)
country_variety = reviews[reviews.country.notnull() & reviews.variety.notnull()][['country', 'variety']]
country_variety = country_variety.apply(lambda srs: srs.country + " - " + srs.variety, axis=1)
check_q7(country_variety.value_counts())