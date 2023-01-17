import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
check_q1(reviews.points.median())
check_q2(reviews.country.unique())
check_q3(reviews.country.value_counts())
review_price_median = reviews.price.median()
check_q4(reviews.price.map(lambda p: p - review_price_median))
def remedian_price(srs):
    srs.price = srs.price - review_price_median
    return srs

check_q5(reviews.apply(remedian_price,axis='columns'))
check_q6(reviews.loc[(reviews.points / reviews.price).idxmax()].title)
# Your code here
# Your code here