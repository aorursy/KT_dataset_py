import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
check_q1(reviews.points.median())
# Your code here
check_q2(reviews.country.unique())
# Your code here
check_q3(reviews.country.value_counts())
# Your code here
median_price = reviews.price.median()
check_q4(reviews.price.map(lambda v: v - median_price))
# Your code here
#reviews.iloc[reviews.points.idxmax(skipna=True)]
#reviews.points.idxmax() / reviews.price.idxmax()
#check_q5(reviews.loc[(reviews.points / reviews.price).idxmax()])
median_price = reviews.price.median()
check_q5(reviews.price.apply(lambda v: v - median_price))
# Your code here
tropical_wine = reviews.description.map(lambda r: "tropical" in r).value_counts()
fruity_wine = reviews.description.map(lambda r: "fruity" in r).value_counts()
check_q7(pd.Series([tropical_wine[True], fruity_wine[True]], index=['tropical', 'fruity']))
# Your code here
ans = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
ans = ans.apply(lambda srs: srs.country + " - " + srs.variety, axis='columns')
check_q8(ans.value_counts())