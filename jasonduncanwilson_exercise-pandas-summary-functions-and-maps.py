import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
reviews.points.median()
#answer_q2()
#check_q2()
# Get list of unique values in a column
reviews.country.unique()
# Count records per unique values in a column
reviews.country.value_counts()
# Calculate difference of prices from the median price
median_price = reviews.price.median()
reviews.price.map(lambda v: v - median_price)
median_price = reviews.price.median()
reviews.price.apply(lambda v: v - median_price)
reviews.loc[(reviews.points / reviews.price).idxmax()].title
tropical_wine = reviews.description.map(lambda r: "tropical" in r).value_counts()
fruity_wine = reviews.description.map(lambda r: "fruity" in r).value_counts()
pd.Series([tropical_wine[True], fruity_wine[True]], index=['tropical', 'fruity'])
ans = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
ans = ans.apply(lambda srs: srs.country + " - " + srs.variety, axis="columns")
ans.value_counts()