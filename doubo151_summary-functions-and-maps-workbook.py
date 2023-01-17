import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
E1 = reviews.points.median()
check_q1(E1)
# Your code here
E2 = reviews.country.unique()
check_q2(E2)
# Your code here
E3 = reviews.country.value_counts()
check_q3(E3)
# Your code here
price_median = reviews.price.median()
E4 = reviews.price.map(lambda p: p - price_median)
check_q4(E4)
# Your code here
E5 = reviews.loc[(reviews.points / reviews.price).idxmax()].title
check_q6(E5)
# Your code here
tropical_wine = reviews.description.map(lambda r: 'tropical' in r).value_counts()
fruity_wine = reviews.description.map(lambda r: 'fruity' in r).value_counts()
E6 = pd.Series([tropical_wine[True], fruity_wine[True]], index=['tropical', 'fruity'])
check_q7(E6)
# Your code here
ans = reviews.loc[reviews.country.notnull() & reviews.variety.notnull()]
ans = ans.apply(lambda srs: srs.country + ' - ' + srs.variety, axis=1)
E7 = ans.value_counts()
check_q8(E7)