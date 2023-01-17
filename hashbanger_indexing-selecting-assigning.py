import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
reviews.description
#or reviews['description']
reviews.description[0]
reviews.iloc[0]
#reviews.iloc[[0], :]
reviews.iloc[0:10, 1]
reviews.iloc[[1,2,3,5,8], :]
reviews.loc[[0,1,10,100], ['country', 'province', 'region_1','region_2']]
#reviews.iloc[[0,1,10,100], [0,5,6,7]]
reviews.loc[0:100, ['country','variety']]
reviews.loc[reviews.country == 'Italy']
reviews.loc[reviews.region_2.notnull()]
reviews.points
reviews.loc[:1000, 'points']
#reviews.points[:1000]
reviews.iloc[-1000:,3]
#reviews.loc[-1000:, 'points']
reviews.points.loc[reviews.country == 'Italy']
#reviews[reviews.country == "Italy"].points
reviews.loc[reviews.country.isin(['France','Italy']) & (reviews.points >= 90)]