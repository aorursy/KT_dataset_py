import pandas as pd
pd.set_option('max_rows', 5)
import numpy as np
from learntools.advanced_pandas.summary_functions_maps import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
reviews['points'].median()
reviews['country'].unique()
check_q2(reviews['country'].unique())
reviews['country'].value_counts()
check_q3(reviews['country'].value_counts())
median = reviews['price'].median()
reviews['price'] - median
check_q4(reviews['price'] - median)
check_q4(reviews['price'].map(lambda x :x - median))
reviews.iloc[(reviews['points'] / reviews['price']).idxmax()]
reviews.loc[(reviews.points / reviews.price).idxmax()].title
tropical = reviews['description'].map(lambda x : True if 'tropical' in x else False).value_counts()[True]
fruity = reviews['description'].map(lambda x : True if 'fruity' in x else False).value_counts()[True]
pd.Series([tropical, fruity], index = ['tropical', 'fruity'])
filtered = reviews[reviews['country'].notnull() & reviews['variety'].notnull()]
(filtered['country'] + '-' + filtered['variety']).value_counts()