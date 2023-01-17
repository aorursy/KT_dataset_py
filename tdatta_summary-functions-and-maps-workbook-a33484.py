import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
reviews.points.median()
reviews.country.unique()
reviews.country.value_counts()
reviews_median_price = reviews.price.median()
reviews.price.map(lambda x : x - reviews_median_price)

reviews_median_price = reviews.price.median()
def remap_price(srs):
    srs.price = srs.price - reviews_median_price
    return srs
reviews.apply(remap_price, axis = 'columns')
def points_to_price(srs):
    srs.points_to_price = srs.points/srs.price
    return srs
reviews.apply(points_to_price, axis = 'columns')





tropical_wine = reviews.description.map(lambda r: "tropical" in r).value_counts()
fruity_wine = reviews.description.map(lambda r: "fruity" in r).value_counts()
pd.Series([tropical_wine[True], fruity_wine[True]], index=['tropical', 'fruity'])

df = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
df = df.apply(lambda srs: srs.country + " - " + srs.variety, axis='columns')
df.value_counts()