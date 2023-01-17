import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
reviews.info()
# Your code here
reviews['points'].median()
check_q1(reviews['points'].median())
# Your code here
reviews['country'].unique()
check_q2(reviews['country'].unique())
# Your code here
reviews['country'].value_counts()
check_q3(reviews['country'].value_counts())
reviews.price.head()
# Your code here
price_median = reviews.price.median()
reviews.price.map(lambda p: p - price_median)
check_q4(reviews.price.map(lambda p: p - price_median))
# Your code here

# Your code here
# Your code here