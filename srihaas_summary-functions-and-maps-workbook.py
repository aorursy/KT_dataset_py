import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
print(check_q1(reviews.points.median()))
print(check_q2(reviews.country.unique()))
print(check_q3(reviews.country.value_counts()))
reviews_price_median = reviews.price.median()
print(check_q4(reviews.price.map(lambda p : p - reviews_price_median)))
k = reviews.loc[(reviews.points / reviews.price).argmax()].title
print(check_q5(k))

# Your code here