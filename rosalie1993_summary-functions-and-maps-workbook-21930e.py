import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
reviews.points.median()
# Your code here
reviews.country.unique()
# Your code here
reviews.country.value_counts()
# Your code here
#print(answer_q4())
b = reviews.price.median()
v = reviews.price.map(lambda a: a - (b))
v
#check_q4(v)
# Your code here
#a = (reviews.points()) / (reviews.price())
#b = reviews.points.map(lambda c: c / reviews.price)
#b
#review
reviews.loc[(reviews.points / reviews.price).idxmax()].title
# Your code here
tropical_wine = reviews.description.map(lambda r: "tropical" in r).value_counts()
fruity_wine = reviews.description.map(lambda r: "fruity" in r).value_counts()
check_q7(pd.Series([tropical_wine[True], fruity_wine[True]], index=['tropical', 'fruity']))
# Your code here
#print(answer_q8())
ans = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
ans = ans.apply(lambda srs: srs.country + " - " + srs.variety, axis='columns')
check_q8(ans.value_counts())
