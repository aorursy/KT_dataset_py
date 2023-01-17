import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(20)
reviews.head()
reviews.points.median()
reviews.country.unique()
reviews.country.value_counts()
review_price_median = reviews.price.median()
reviews.price.map(lambda x: x - review_price_median)

# Your code here
# Your code here