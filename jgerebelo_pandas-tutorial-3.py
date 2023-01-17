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
reviews.points.median()
check_q1(reviews.points.median())
# Your code here
reviews['country'].unique()
check_q2(reviews['country'].unique())
reviews.country.unique()
check_q2(reviews.country.unique())
# Your code here
reviews['country'].value_counts()
check_q3(reviews['country'].value_counts())
# Your code here
median_price = reviews['price'].median()
reviews['price'].map(lambda v: v - median_price)
check_q4(reviews['price'].map(lambda v: v - median_price))
reviews['price'] - reviews['price'].median()
check_q4(reviews['price'] - reviews['price'].median())
# Your code here
reviews['title'].loc[(reviews.points / reviews.price).idxmax()]
check_q5(reviews['title'].loc[(reviews.points / reviews.price).idxmax()])
reviews.loc[(reviews.points / reviews.price).idxmax()].title
check_q5(reviews.loc[(reviews.points / reviews.price).idxmax()].title)
reviews.loc[(reviews.points / reviews.price).idxmax(), 'title']
check_q5(reviews.loc[(reviews.points / reviews.price).idxmax(), 'title'])
# Your code here
tropical_wine = reviews.description.map(lambda r: 'tropical' in r).value_counts()
fruity_wine = reviews.description.map(lambda r: 'fruity' in r).value_counts()
pd.Series([tropical_wine[True], fruity_wine[True]], index = ['tropical', 'fruity'])
check_q6(pd.Series([tropical_wine[True], fruity_wine[True]], index = ['tropical', 'fruity']))
# Your code here
ind = reviews.loc[(reviews.country.notnull()) & (reviews.variety.notnull())]
l_ind = ind.apply(lambda a: a.country + ' - ' + a.variety, axis = 'columns')
l_ind.value_counts()
check_q7(l_ind.value_counts())