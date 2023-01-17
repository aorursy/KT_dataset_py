import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
reviews.tail()
check_q1(pd.DataFrame())
# Your code here
check_q1(reviews['description'])
# Your code here
check_q2(reviews['description'][0])
# Your code here
check_q3(reviews.iloc[0])
# Your code here
check_q4(reviews['description'][0:10])
# Your code here
a5 = reviews.iloc[[1,2,3,5,8],:]
a5
check_q5(a5)
# Your code here
a6 = reviews.loc[[0,1,10,100],['country', 'province', 'region_1', 'region_2']]
check_q6(a6)
# Your code here
a7 = reviews.loc[0:100,['country', 'variety']]
check_q7(a7)
#a7
# Your code here
check_q8(reviews[reviews.country == 'Italy'])
# Your code here
check_q9(reviews[reviews.region_2.notnull()])
# Your code here
check_q10(reviews.points)
reviews.points
# Your code here
a11 = reviews.loc[0:999, 'points']
check_q11(a11)

a11
# Your code here
# Your code here
a12 = reviews.points[-1000:]
check_q12(a12)
#a12
reviews.iloc[-1000:].points
# Your code here

a13=reviews.loc[reviews.country == 'Italy','points']
check_q13(a13)
reviews.country.isin(['Italy','France'])
# Your code here
a14 = reviews.loc[(reviews.country.isin(['Italy','France'])) & (reviews.points >= 90)].country
check_q14(a14)
#a14