import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
# Your code here
reviews['points'].median()
# Your code here
reviews.country.unique()
reviews.country.mode()
med = reviews['price'].median()
reviews.price.map(lambda p: p - med)
highest_points_ratio_index = (reviews.points / reviews.price).idxmax()
reviews.loc[highest_points_ratio_index].title
has_tropical = reviews['description'].map(lambda r: "tropical" in r)
tropical = reviews[has_tropical]
has_fruity = reviews['description'].map(lambda r: "fruity" in r)
fruity = reviews[has_fruity]
s = pd.Series([tropical['title'].count(),fruity['title'].count()],index=['tropical','fruity'])
check_q7(s)
not_null = reviews.dropna(subset=['country', 'variety'])
reviews['new_index'] = reviews['country'] + ' - ' + reviews['variety']
counts = reviews['new_index'].value_counts()
check_q8(counts)