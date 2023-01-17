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
check_q3((reviews.country.value_counts()))
# Your code here
price_median = reviews.price.median()
check_q4(reviews.price.map(lambda x: x - price_median))

# Your code here
check_q5(reviews.title[(reviews.points/reviews.price).idxmax()])

# Your code here
print(reviews.loc[reviews.description.str.contains('tropical')].title.count())
print(reviews.loc[reviews.description.str.contains('fruity')].title.count())
check_q6(pd.Series([reviews.loc[reviews.description.str.contains('tropical')].title.count(),
                   reviews.loc[reviews.description.str.contains('fruity')].title.count()]))
answer_q6()
# Your code here
combined = pd.Series(reviews.country + ' - ' + reviews.variety)
combined = combined.loc[combined.notnull()]
print(combined.value_counts())
answer_q7()
check_q7(combined.value_counts())