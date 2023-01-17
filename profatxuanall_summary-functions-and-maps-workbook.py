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
m = reviews.price.median()
check_q4(reviews.price.map(lambda d: d-m))
# Your code here
check_q6(reviews.loc[(reviews.points / reviews.price).idxmax()].title)
# Your code here
tropical = reviews.description.map(lambda d: 'tropical' in d).sum()
fruity = reviews.description.map(lambda d: 'fruity' in d).sum()
check_q7(pd.Series([tropical, fruity], index=['tropical','fruity']))
# Your code here
result = reviews[reviews.country.notnull() & reviews.variety.notnull()]
check_q8((result.country + ' - ' + result.variety).value_counts())