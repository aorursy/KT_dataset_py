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
reviews.country.describe().top
# Your code here
median = reviews.price.median()
check_q4(reviews.price.map(lambda p: p - median))
# Your code here
_id = (reviews.points / reviews.price).idxmax()
reviews.iloc[_id:_id + 1]
# Your code here
tropical_wines = reviews.description.map(lambda d: 'tropical' in d.lower())

fruity_wines = reviews.description.map(lambda d: 'fruity' in d.lower())

print('Count tropical wines', tropical_wines.sum())
print('Count fruity wines', fruity_wines.sum())
# Your code here
reviews2 = reviews[(reviews.country.notnull()) & (reviews.variety.notnull())]
reviews2['country-variety'] = reviews.apply(lambda r: str(r.country) + '-' + str(r.variety), axis='columns')
reviews2['country-variety'].value_counts()