import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
check_q1(reviews.description)
#reviews.description
# Your code here
check_q2(reviews.description[0])
#reviews.description[0]
# Your code here
reviews.loc[0]
#check_q3(reviews.loc[0])
#check_q3(reviews.iloc[0])
#answer_q3()
# Your code here
#check_q4(reviews.head(10))
check_q4(reviews.iloc[0:10,1])
reviews.iloc[0:10,0]
#answer_q4()
#help(reviews.head)
# Your code here
check_q5(reviews.iloc[[1,2,3,5,8]])
# Your code here
#check_q6(reviews.description[{"country","province","region_1","region_2"}])
#check_q6(reviews.loc[[0,1,10,100], [`country`, `province`, `region_1`, `region_2`]])
check_q6(reviews.loc[[0,1,10,100], ['country', 'province', 'region_1', 'region_2']])
#answer_q6()
# Your code here
check_q7(reviews.loc[0:100,['country','variety']])
# Your code here
#answer_q8()
check_q8(reviews.loc[reviews.country=='Italy'])
# Your code here
#check_q9(reviews.loc[reviews.region_2 is not Null])
#answer_q9()
check_q9(reviews.loc[reviews.region_2.notnull()])
# Your code here
#check_q10(reviews.description[points])
#answer_q10()
check_q10(reviews.points)
# Your code here
#check_q11(reviews.points[0:1000])
#answer_q11()
check_q11(reviews.loc[:1000,'points'])
# Your code here
#check_q12(reviews.iloc[-1000:-1,3])
#answer_q12()
check_q12(reviews.points[-1000:])
# Your code here
#answer_q13()
check_q13(reviews[reviews.country=='Italy'].points)
# Your code here
# Your code here
#answer_q14()
#check_q14(reviews[reviews.country.isin(['Italy''France'])&(reviews.points >= 90)].country)
check_q14(reviews[reviews.country.isin(["Italy", "France"]) & (reviews.points >= 90)].country)
#answer_q14()