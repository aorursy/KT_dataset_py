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
reviews['description']
check_q1(pd.DataFrame())
# Your code here
reviews['description'][0]
check_q2(pd.DataFrame())
# Your code here
reviews.iloc[0]
check_q3(pd.DataFrame())
# Your code here
reviews.iloc[[0,1,2,3,4,5,6,7,8,9],1]
check_q4(pd.DataFrame())
# Your code here
reviews.iloc[[1,2,3,5,8],:]
check_q5(pd.DataFrame())
# Your code here
reviews.loc[[0,1,10,100],['country','province','region_1','region_2']]
check_q6(pd.DataFrame())
# Your code here
reviews.loc[0:100,['country','variety']]
# Your code here
reviews.loc[reviews.country=='Italy']
# Your code here
reviews.loc[reviews.region_2!='NaN']
# Your code here
reviews.points
# Your code here
reviews.points[0:1000]
# Your code here
reviews.points[-1000:]
reviews.loc[(reviews.country=='Italy')&(reviews.points)]
reviews.loc[((reviews.country == 'Italy')|(reviews.country=='France')) & (reviews.points >= 90)]