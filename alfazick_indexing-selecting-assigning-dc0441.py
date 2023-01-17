import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
ans1 = reviews.description
check_q1(ans1)
# Your code here
ans2 = reviews['description'][0]
check_q2(ans2)
# Your code here
ans3 = reviews.loc[0]
check_q3(ans3)
# Your code here
ans4 = reviews['description'].iloc[0:10]
ans4
check_q4(ans4)
# Your code here
ans5 = reviews.iloc[[1,2,3,5,8]]
ans5
check_q5(ans5)
# Your code here
ans6 = reviews[['country','province','region_1','region_2']].iloc[[0,1,10,100]]
check_q6(ans6)
# Your code here
ans7 = reviews[['country','variety']].loc[0:100]
ans7
check_q7(ans7)
# Your code here
ans8 = reviews[reviews.country == 'Italy']
ans8
check_q8(ans8)
# Your code here
ans9 = reviews[reviews.region_2.notnull()]
ans9
check_q9(ans9)
# Your code here
ans10 = reviews.points
ans10
check_q10(ans10)
# Your code here
ans11 = reviews.points.loc[0:1000]
ans11
check_q11(ans11)
# Your code here
ans12 = reviews.points.tail(1000)
ans12
check_q12(ans12)
# Your code here
ans13 = reviews.points[reviews.country == 'Italy']
ans13
check_q13(ans13)
# Your code here