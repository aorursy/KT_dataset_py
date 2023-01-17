import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
check_q2(reviews.description)
check_q2(reviews.description[0])
check_q3(reviews.iloc[0,:])
check_q4( reviews.iloc[:10,1])
# Your code here
dataf = reviews.iloc[[1,2,3,5,8],:]
check_q5(dataf)
# Your code here
dataf = reviews.loc[[0,1,10,100],['country','province','region_1','region_2']]
check_q6(dataf)
# Your code here
check_q7(reviews.loc[:100,['country','variety']])
# Your code here
check_q8(reviews.query('country == "Italy"'))
result = reviews.loc[reviews.region_2.notnull()]
check_q9(result)
# Your code here
check_q10(reviews.points)
# Your code here
check_q11(reviews.points[:1000])
# Your code here
check_q12(reviews.points[-1000:])
# Your code here
check_q13(reviews.points[reviews.country == 'Italy'])
# Your code here
r = reviews.country[(reviews.country.isin(['Italy','France'])) & (reviews.points >= 90)]
check_q14(r)