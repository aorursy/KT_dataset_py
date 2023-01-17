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
reviews['points'].median()
check_q1(reviews['points'].median())
# Your code here
reviews['country'].unique()
check_q2(reviews['country'].unique())
# Your code here
reviews['country'].value_counts()
check_q3(reviews['country'].value_counts())
# Your code here
median_price = reviews['price'].median()
reviews['price'].map(lambda p: p - median_price)
check_q4(reviews['price'].map(lambda p: p - median_price))
# Your code here
reviews.loc[(reviews['points'] / reviews['price']).idxmax(),'title']

check_q5(reviews.loc[(reviews['points'] / reviews['price']).idxmax(),'title'])
# Your code here
tropical = reviews['description'].map(lambda desc: 'tropical' in desc)
tropical_counts = tropical.value_counts()
fruity = reviews['description'].map(lambda desc: 'fruity' in desc)
fruity_counts = fruity.value_counts()
pd.Series([tropical_counts[True], fruity_counts[True] ], index=['tropical', 'fruity'])
check_q6(pd.Series([tropical_counts[True], fruity_counts[True] ], index=['tropical', 'fruity']))
# Your code here
country_variety = reviews.loc[(reviews['country'].notnull() & reviews['variety'].notnull())]
country_variety
df_country_variety = country_variety.apply(lambda srs: srs['country'] + " - " + srs['variety'], axis='columns')
df_country_variety.value_counts()
check_q7(df_country_variety.value_counts())