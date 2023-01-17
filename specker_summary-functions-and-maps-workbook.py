import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
check_q1(reviews['points'].median())
check_q2(reviews['country'].unique())
check_q3(reviews['country'].value_counts())
reviews_median = reviews['price'].median()
reviews['price'].map(lambda x: x - reviews_median)

reviews['ratio'] = reviews['points']/reviews['price']
reviews['ratio'].idxmax()

def count(data):
    a = 0
    data.map(lambda x: True if x == 'tropical' else False)
    

data = reviews['description']
count(data)
# Your code here