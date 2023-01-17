import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
check_q1(reviews['points'].median())
# Your code here
check_q2(reviews['country'].drop_duplicates())
# Your code here
check_q3(reviews['country'].value_counts())
# Your code here
price_median = reviews['price'].median()
check_q4(reviews['price'].map(lambda p: p - price_median))
# Your code here
highest_points_ratio_index = (reviews.points / reviews.price).idxmax()
# checks are confused
check_q6(reviews.loc[highest_points_ratio_index].title) 
# Your code here
has_tropical = reviews['description'].map(lambda r: "tropical" in r)
tropical = reviews[has_tropical]
has_fruity = reviews['description'].map(lambda r: "fruity" in r)
fruity = reviews[has_fruity]
res = pd.Series([tropical['title'].count(), fruity['title'].count()], index = ['tropical', 'fruity'])
check_q7(res)
# Your code here
not_null = reviews.dropna(subset=['country', 'variety'])
reviews['new_index'] = reviews['country'] + ' - ' + reviews['variety']
counts = reviews['new_index'].value_counts()
check_q8(counts)