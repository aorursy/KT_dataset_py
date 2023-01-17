import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
reviews['description'].head()
# Your code here
reviews.description[0]
# Your code here
reviews.loc[0, :]
#reviews.iloc[0]
reviews.iloc[0:10, 1]
reviews.iloc[[1,2,3,5,8]]
# Your code here
reviews.loc[[0, 1, 10, 100], ['country', 'province', 'region_1', 'region_2']]
# Your code here
reviews.loc[0:100, ['country', 'variety']]
# Your code here
reviews.loc[reviews.country == 'Italy']
# Your code here
reviews.loc[reviews.region_2.notnull()]
# Your code here
#reviews['points'].head()
reviews.points
#reviews.points[0:1000]
reviews.loc[0:1000, 'points']
# Your code here
reviews.loc[-1000:, 'points']
#reviews[reviews.country == "Italy"].points
reviews.points.loc[reviews.country == 'Italy']
# reviews.[reviews.country.isin(['Italy', 'France']) & (reviews.points >= 90)].country
reviews.country.loc[reviews.country.isin(['Italy', 'France']) & (reviews.points >= 90)]