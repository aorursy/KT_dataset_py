import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
reviews.description
reviews.description[0]
reviews['description'][0]
reviews.iloc[0]
reviews.loc[0]
reviews.iloc[0:1,]# First row with all columns
reviews.iloc[0:1, [0,1,2]] # First row with 3 columns
reviews.iloc[0:10,1]
answer_q4()
reviews.iloc[0:10,1]
reviews.iloc[[1,2,3,5,8],]
reviews.iloc[[0,1,10,100],[0,5,6,7]]
reviews.iloc[0:100,[0,11]]
reviews.country == 'Italy'
reviews.loc[reviews.country == 'Italy']
reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)]
reviews.loc[(reviews.region_2.notnull())]
reviews.loc[(reviews.points)]
reviews.iloc[0:,3]
reviews.head()
# Your code here
# Your code here
# Your code here
# Your code here