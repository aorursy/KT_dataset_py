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
check_q1(reviews.points.median())
# Your code here
check_q2(reviews.country.unique())
# Your code here
# correct answer: reviews.country.mode()
check_q3(reviews.country.value_counts())
# Your code here
med = reviews.price.median()
check_q4(reviews.price.map(lambda v: v - med))
# Your code here
idxmax = (reviews.points / reviews.price).idxmax()
check_q5(reviews.title[idxmax])
# Your code here
tropical = reviews.description.map(lambda d: "tropical" in d).value_counts()
fruity = reviews.description.map(lambda d: "fruity" in d).value_counts()
result = pd.Series([tropical[True], fruity[True]], index=['tropical', 'fruity'])
check_q6(result)
# Your code here
result = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
ans = result.apply(lambda row: row.country + " - " + row.variety, axis='columns')
check_q7(ans.value_counts())