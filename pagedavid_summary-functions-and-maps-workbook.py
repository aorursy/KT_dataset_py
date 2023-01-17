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
median = reviews.price.median()
check_q4(reviews.price.map(lambda x: x - median))
# Your code here
median = reviews.price.median()
check_q4(reviews.price.apply(lambda x: x - median))
# Your code here
check_q5(reviews.loc[(reviews.points / reviews.price).idxmax()].title)
# Your code here
fruity = reviews.description.map(lambda x: 'fruity' in x).value_counts()[True]
tropical = reviews.description.map(lambda x: 'tropical' in x).value_counts()[True]
pd.Series([tropical, fruity], index=['tropical', 'fruity'])
# Your code here
ans = reviews.loc[reviews.country.notnull() & reviews.variety.notnull()]
ans = ans.apply(lambda x: x.country + " - " + x.variety, axis='columns')
check_q7(ans.value_counts())