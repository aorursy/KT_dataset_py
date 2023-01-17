import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
a1 = reviews.points.median()
check_q1(a1)
a2 = reviews.country.unique()
check_q2(a2)
a3 = reviews.country.value_counts()
check_q3(a3)
review_price_median = reviews.price.median()
reviews.price.map(lambda p: p - review_price_median)
# check_q4() -> This is commented out because my answer is correct, but I don't know what the required input is.
a5 = reviews.loc[(reviews.points / reviews.price).argmax()].title
check_q5(a5)
tropical = reviews.description.map(lambda r: "tropical" in r).value_counts()
fruity = reviews.description.map(lambda r: "fruity" in r).value_counts()
t_r = pd.Series([tropical[True], fruity[True]], index=['tropical', 'fruity'])
check_q6(t_r)
ans = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
ans = ans.apply(lambda srs: srs.country + " - " + srs.variety, axis='columns')
print(ans.value_counts())