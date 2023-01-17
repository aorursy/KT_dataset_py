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
reviews['description'].iloc[0]
reviews.iloc[0]
reviews['description'].iloc[0:10:1]
reviews['description'].iloc[[1,3,7,8]]
reviews[['country', 'province', 'region_1','region_2']].iloc[[0,1,10,100]]
reviews[['country', 'variety']].iloc[0:101]
reviews[reviews['country']=='Italy']
reviews[pd.isnull(reviews['region_2'])==False]
reviews['points']
reviews['points'].iloc[0:1001]
reviews.tail(1000)['points']
reviews[reviews['country']=='Italy']['points']
reviews[reviews['points']>=90]['country']