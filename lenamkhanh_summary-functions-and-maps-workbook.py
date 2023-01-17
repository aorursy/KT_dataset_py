import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
print(reviews.points.mean())
check_q1(reviews.points.median())
# Your code here
print(reviews.country.unique())
check_q2(reviews.country.unique())
# Your code here
print(reviews.country.value_counts())
check_q3(reviews.country.value_counts())
# Your code here
review_price_median = reviews.price.median()
reviews.price.map(lambda p: p - review_price_median)
check_q4(reviews.price)
# Your code here
def remean_price(srs):
    srs.price = srs.price - review_price_mean
    return srs

reviews.apply(remean_price, axis='columns')
check_q4(reviews.price)
# Your code here
print(reviews.points / reviews.price)
idxmax = (reviews.points / reviews.price).idxmax()
print(idxmax)
check_q5(reviews.title[idxmax])
# Your code here
a = reviews.description.str.contains('tropical').value_counts()
print(a)
b = reviews.description.str.contains('fruity').value_counts()
print(b)
series = pd.Series([a[True],b[True]],index =['tropical','fruity'])
check_q6(series)
# Your code here
result = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
ans = result.apply(lambda row: row.country + " - " + row.variety, axis='columns')
check_q7(ans.value_counts())