import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
desc = reviews['description']
check_q1(desc)
desc_1 = desc.loc[0]
check_q2(desc_1)
rev_1 = reviews.loc[0]
check_q3(rev_1)
rev_5 = reviews.iloc[:10,1]
rev_5
check_q4(rev_5)
rev_2 = reviews.iloc[[1,2,3,5,8], :]
check_q5(rev_2)
rev_3 = reviews.loc[[0,1,10,100],['country', 'province', 'region_1', 'region_2']]
check_q6(rev_3)
rev_4 = reviews.loc[:100, ['country', 'variety']]
check_q7(rev_4)
rev_5 = reviews[reviews.country=='Italy']
check_q8(rev_5)
rev_6 = reviews[reviews.region_2.notna()]
check_q9(rev_6)
rev_7 = reviews.loc[:,'points']
check_q10(rev_7)
rev_8 = reviews.loc[:1000, 'points']
check_q11(rev_8)

rev_9 = reviews.loc[128971:, 'points']
check_q12(rev_9)
rev_10 = reviews.loc[:, 'points'][reviews.country=='Italy']
check_q13(rev_10)
rev_11 = reviews.loc[:, :][reviews.points>=90]
rev_12 = rev_11.country[rev_11.country.isin(['Italy','France'])]
check_q14(rev_12)