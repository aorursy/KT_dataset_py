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
reviews.points.median()

# Your code here
reviews.country.unique()
# Your code here
reviews.country.value_counts()
median_price = reviews.price.median()
reviews.price.map(lambda v: v - median_price)
# Your code here
review_price_median = reviews.price.median()
reviews.price - review_price_median
# Your code here
reviews.head()

reviews.title.loc[(reviews.points/ reviews.price).idxmax()]
tropical_wine = reviews.description.map(lambda r: "tropical" in r).value_counts()
fruity_wine = reviews.description.map(lambda r: "fruity" in r).value_counts()
pd.Series([tropical_wine[True], fruity_wine[True]], index=['tropical', 'fruity'])
ans = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
ans = ans.apply(lambda srs: srs.country + " - " + srs.variety, axis='columns')
ans.value_counts()