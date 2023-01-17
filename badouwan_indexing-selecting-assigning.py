import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
q1 = reviews['description']
check_q1(q1)
q2 = reviews['description'][0]
check_q2(q2)
q3 = reviews.iloc[0]
check_q3(q3)
q4 = reviews['description'].head(10)
check_q4(q4)
q5 = reviews.iloc[[1,2,3,5,8]]
check_q5(q5)
q6 = reviews[['country','province','region_1','region_2']].iloc[[0,1,10,100]]
check_q6(q6)
q7 = reviews[['country','variety']].iloc[0:101]
check_q7(q7)
q8 = reviews[reviews.country=='Italy']
check_q8(q8)
q9 = reviews[reviews.region_2.notnull()]
check_q9(q9)
q10 = reviews['points']
check_q10(q10)
q11 = reviews['points'].iloc[0:1001]
check_q11(q11)
q12 = reviews['points'].iloc[-1000:]
check_q12(q12)
q13 = reviews[reviews['country']=='Italy']['points']
check_q13(q13)
q14 = reviews.loc[(reviews.country=='France') | (reviews.country=='Italy')].loc[reviews.points>=90]['country']
check_q14(q14)