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
check_q1(reviews.loc[:,'description'])
# Your code here
check_q2(reviews.iloc[0][1])
# Your code here
check_q3(reviews.iloc[0])
# Your code
check_q4[reviews.iloc[:10,2]]
# Your code here
reviews.iloc[[1,2,3,5,8]]
# Your code here
check_q6(reviews.loc[[0,1,10,100], ['country', 'province', 'region_1', 'region_2']])
# Your code here
check_q7(reviews.loc[:100, ['country', 'variety']])
# Your code here
answer_q8()

# Your code here
check_q9(reviews.loc[reviews.region_2.notnull()])
#answer_q9()
# Your code here
check_q10(reviews.loc[:, 'points'])
# Your code here
check_q11(reviews.loc[:1000, 'points'])
# Your code here
check_q12(reviews.iloc[-1000:, 3])
#reviews.loc[-1000:, 'points']
# Your code here
#reviews.loc[:, 'points' : reviews.country == 'Italy']
answer_q13()
# Your code here
#reviews[reviews.points >= 90].country = 'France' or 'Italy'
answer_q14()
#reviews.country == 'France' & 'Italy'
#reviews[(reviews.country == ["Italy", "France"]) & (reviews.points >= 90)].country