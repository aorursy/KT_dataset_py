import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
reviews.filter(items=["description"])
reviews.at[0,"description"]
reviews.loc[0]
reviews.loc[0:9].filter(items=["description"])
reviews.loc[[1,2,3,5,8]]
reviews.loc[[0,1,10,100]].filter(["country","province","region_1","region_2"])
reviews.loc[0:99].filter(["country","variety"])
reviews.loc[reviews.country == 'Italy']
reviews.loc[reviews.region_2 != 'NaN']
reviews.filter(items=["points"])
reviews.points.loc[0:999]
reviews.points.loc[reviews.shape[0]-1000:reviews.shape[0]]
reviews.points.loc[reviews.country == 'Italy']
reviews[reviews.country.isin(["Italy", "France"]) & (reviews.points >= 90)].country