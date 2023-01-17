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
answer = reviews.loc[0:9,'description']
q4 = pd.Series(answer)
print(q4)
check_q4(q4)
# Your code here
q5 = reviews.iloc[[1,2,3,5,8],:]
#print(q5)
check_q5(q5)
# Your code here
q6 = reviews.loc[[0,1,10,100],["country","province","region_1","region_2"]]
check_q6(q6)
# Your code here
q7 = reviews.loc[0:99,["country","variety"]]
#this should be the correct code. 
check_q7(q7)
# Your code here
q8 = reviews[reviews.country == "Italy"]
check_q8(q8)

# Your code here
q9 = reviews[reviews.region_2.notnull()]
check_q9(q9)
# Your code here
q10 = reviews.points
check_q10(q10)
# Your code here
q11 = reviews.points[0:999]
check_q11(q11)
# Your code here
q12 = reviews.points[-1000:]
check_q12(q12)

# Your code here
q13 = reviews.points[reviews.country == "Italy"]
check_q13(q13)

# Your code here
q14 = reviews.country[(reviews.country.isin(['Italy','France'])) & (reviews.points  >=90) ]

check_q14(q14)