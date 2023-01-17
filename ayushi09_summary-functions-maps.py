import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(20)
reviews.head()
check_q1(reviews.points.median())
# Your code here
# Your code here
check_q2(reviews.country.unique())
# Your code here
check_q3(reviews.country.value_counts())
# Your code here
reviews_price_column=reviews.price.median()
check_q4(reviews.price - reviews_price_column)
reviews['Res'] = reviews['points']/reviews['price']
#reviews.groupby('title')['Res'].idxmax()
we = reviews.loc[reviews['Res'].idxmax()]
check_q5(we['title'])

# Your code here
pd.value_counts(reviews['description'].values, sort=False)
# Your code here
check_q7((reviews.country + " - " + reviews.variety).value_counts())

