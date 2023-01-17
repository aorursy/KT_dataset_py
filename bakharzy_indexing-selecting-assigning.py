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
reviews.description

# Your code here
reviews['description'][0]
check_q2(reviews['description'][0])
# Your code here
reviews.iloc[0]
check_q3(reviews.iloc[0])
# Your code here
reviews.iloc[0:10,1]
check_q4(reviews.iloc[0:10,1])
# Your code here
reviews.iloc[[1,2,3,5,8]]
check_q5(reviews.iloc[[1,2,3,5,8]])
# Your code here
reviews.iloc[[0,1,10,100],[0,5,6,7]]
reviews.loc[[0,1,10,100],['country','province','region_1','region_2']]
check_q6(reviews.loc[[0,1,10,100],['country','province','region_1','region_2']])
# Your code here
reviews.iloc[0:101,[0,11]]
reviews.loc[0:100,['country','variety']]
check_q7(reviews.loc[0:100,['country','variety']])

# Your code here
reviews.loc[reviews.country=='Italy']
check_q8(reviews.loc[reviews.country=='Italy'])
# Your code here
reviews.loc[reviews.region_2.notnull()]
check_q9(reviews.loc[reviews.region_2.notnull()])
# Your code here
reviews.loc[:,'points']
reviews.points
check_q10(reviews.points)

# Your code here
reviews.loc[0:1000,'points']
check_q11(reviews.loc[0:1000,'points'])

# Your code here
reviews.loc[-1000:,'points']

# Your code here
reviews[reviews.country=='Italy'].points
check_q13(reviews[reviews.country=='Italy'].points)
# Your code here
reviews.loc[(reviews.country.isin(['Italy', 'France'])) & (reviews.points >= 90)].country
check_q14(reviews.loc[(reviews.country.isin(['Italy', 'France'])) & (reviews.points >= 90)].country)
