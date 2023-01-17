import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
reviews['description']
reviews['description'][0]
reviews.loc[0,:]
reviews['description'][:10]
reviews.loc[:9,'description']
reviews.loc[[1,2,3,5,8],:]
reviews.loc[[0,1,10,100], ['country', 'province', 'region_1', 'region_2']]
reviews.loc[:99, ['country', 'variety']]
reviews[reviews.country == 'Italy']
reviews[reviews.region_2.notnull()]
reviews['points']
reviews.loc[:999,['points']]
reviews.loc[:,['points']][-1000:]
reviews[reviews['country'] == 'Italy']['points']
reviews[(reviews['points'] >= 90) & reviews['country'].isin(['France', 'Italy'])]