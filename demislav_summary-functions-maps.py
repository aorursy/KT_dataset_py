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
# Your code here
reviews.country.unique()
# Your code here
reviews.country.value_counts()
# Your code here
reviews['price'] = reviews.price-reviews.price.median()

# Your code here
ratio = reviews['points']/reviews['price']
ratio.idxmax()
reviews.loc[64590, 'title']
# Your code here
desc = pd.Series(reviews.description)
desc_tr = desc.str.count('tropical').value_counts()[1:].sum()
desc_fr = desc.str.count('fruity').value_counts()[1:].sum()
pd.Series([desc_tr, desc_fr],index=['tropical', 'fruity'])
check_q6(pd.Series([desc_tr, desc_fr],index=['tropical', 'fruity']))
# Your code here
dat = reviews[['country', 'variety']].dropna()
srs = dat.country+' - '+dat.variety
check_q7(srs.value_counts())