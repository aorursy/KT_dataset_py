import pandas as pd
pd.set_option('max_rows', 10)
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
check_q4(reviews.price.map(lambda p: p - median_price))
# Your code here
#ptpr = np.nanmax((reviews.points / reviews.price))
#reviews.loc[(reviews.points / reviews.price) == ptpr].title
check_q6(reviews.loc[(reviews.points / reviews.price).idxmax()].title)
# Your code here
tropical_count = reviews.description.map(lambda d: True if 'tropical' in d else False).value_counts()[True]
fruity_count = reviews.description.map(lambda d: True if 'fruity' in d else False).value_counts()[True]
check_q7(pd.Series([tropical_count, fruity_count], index=['tropical', 'fruity']))
# Your code here
df = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
s = df.country + ' - ' + df.variety
check_q8(s.value_counts())