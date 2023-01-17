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
check_q1(reviews.description)
# Your code here
# reviews['description'].iloc[0]
check_q2(reviews['description'].iloc[0])
# Your code here
check_q3(reviews.iloc[0,:])
# Your code here
check_q4(reviews['description'].iloc[0:10])
# answer_q4()
# Your code here
check_q5(reviews.iloc[[1,2,3,5,8],:])
# Your code here
check_q6(reviews[['country', 'province', 'region_1', 'region_2']].iloc[[0,1,10,100],:])
# Your code here
# Actually it should be 100 if you want the first 100 records.
check_q7(reviews[['country', 'variety']].iloc[0:101])
# reviews.loc[0:100, ['country', 'variety']]
# answer_q7()
# Your code here
# reviews[reviews.country == 'Italy']
check_q8(reviews[reviews.country == 'Italy'])
# Your code here
check_q9(reviews[reviews.region_2.notnull()])
# Your code here
check_q10(reviews.points)
# Your code here
check_q11(reviews.points.iloc[0:1000])
# Your code here
reviews.points.iloc[-1000:]
check_q12(reviews.points.iloc[-1000:])
# Your code here
reviews[reviews.country == 'Italy'].points
check_q13(reviews[reviews.country == 'Italy'].points)
# Your code here
answer = reviews[reviews.country.isin(['France', 'Italy'])][reviews.points >= 90].country
check_q14(answer)