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
check_q2(reviews.country)
# Your code here
check_q3(reviews.country.mode())
answer_q3()
reviews.country.value_counts()
# Your code here
answer_q4()
median_price = reviews.price.median()
median_price
reviews.price.map(lambda v:v - median_price)
# Your code here
answer_q5()
# Your code here
# Your code here