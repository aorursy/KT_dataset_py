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
#print(answer_q1())
reviews.points.median()
# Your code here
#print(answer_q2())
reviews.country.unique()
# Your code here
#print(answer_q3())
reviews.country.value_counts()
# Your code here
#print (answer_q4())
median_price = reviews.price.median()
reviews.price.map(lambda v: v - median_price)
# Your code here
#reviews.title.idxmax[]
#print (answer_q5())
reviews.loc[(reviews.points / reviews.price).idxmax()].title
# Your code here
#print (answer_q6())

tropical_wine = reviews.description.map(lambda r: "tropical" in r).value_counts()
fruity_wine = reviews.description.map(lambda r: "fruity" in r).value_counts()
pd.Series([tropical_wine[True], fruity_wine[True]], index = ['tropical', 'fruity'])
#print (answer_q7())
# Your code here
#print (answer_q7())
#这个还不太理解。。。。

ans = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
ans = ans.apply(lambda srs: srs.country + " - " + srs.variety, axis='columns')
ans.value_counts()