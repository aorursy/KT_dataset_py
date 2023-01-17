import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
d1=reviews.points.median()
check_q1(d1)
# Your code here
d2=reviews.country.unique()
check_q2(d2)
# Your code here
d3=reviews.country.value_counts()
check_q3(d3)
# Your code here
review_price_median=reviews.price.median()
d4=reviews.price.apply(lambda p: p - review_price_median)
check_q4(d4)
# Your code here
bargain = reviews.points / reviews.price
d5=reviews.loc[bargain.idxmax()].title
#shifted question check
check_q6(d5)

# Your code here
tropical = reviews.description.map(lambda r: "tropical" in r).value_counts()
fruity = reviews.description.map(lambda r: "fruity" in r).value_counts()
d6=pd.Series([tropical[True], fruity[True]], index=['tropical', 'fruity'])

check_q7(d6)
# Your code here
df=reviews.loc[reviews.country.notnull() & reviews.variety.notnull()]
df=df.apply(lambda s: s.country + " - " + s.variety, axis='columns')
d7=df.value_counts()
check_q8(d7)