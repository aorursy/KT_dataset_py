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
q1 = reviews.description
check_q1(q1)
# Your code here
q2 = q1[0]
check_q2(q2)
# Your code here
q3 = reviews.iloc[0]
check_q3(q3)
# Your code here
q4 = q1.iloc[:10]
check_q4(q4)
# Your code here
q5 = reviews.iloc[[1,2,3,5,8]]
check_q5(q5)
# Your code here
q6 = reviews.loc[[0,1,10,100], ['country', 'province', 'region_1', 'region_2']]
check_q6(q6)
# Your code here
q7 = reviews.loc[:100, ['country', 'variety']]

check_q7(q7)
len(q7)
# Your code here
q8 = reviews[reviews.country=='Italy']
check_q8(q8)
# Your code here
q9 = reviews[reviews.region_2.isna()==False]
q9
check_q9(q9)
# Your code here
q10 = reviews['points']
check_q10(q10)
# Your code here
q11 = reviews.loc[:1001, 'points']
check_q11(q11)
# Your code here
q12 = reviews['points'].tail(1000)
check_q12(q12)
# Your code here
q13 = reviews[reviews['country'] == 'Italy'].points
check_q13(q13)
# Your code here
q14 = reviews[((reviews['country'] == 'Italy')&(reviews['points'] >= 90)) | ((reviews['country'] == 'France')&(reviews['points'] >= 90))]['country']
q14
check_q14(q14)