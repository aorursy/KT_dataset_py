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
reviews.description
# Your code here
reviews.description[0]
check_q2(reviews.description[0])
# Your code here
reviews.iloc[0]
check_q3(reviews.iloc[0])
# Your code here
reviews['description'][:10]
check_q4(reviews['description'][:10])
# Your code here
reviews.iloc[[1,2,3,5,8],:]
check_q5(reviews.iloc[[1,2,3,5,8],:])
# Your code here
reviews.loc[[0, 1, 10, 100], ['country', 'province', 'region_1', 'region_2']]
check_q6(reviews.loc[[0, 1, 10, 100], ['country', 'province', 'region_1', 'region_2']])
# Your code here
reviews.loc[:99, ['country', 'variety']]
check_q7(reviews.loc[:99, ['country', 'variety']])
# Your code here
reviews.loc[reviews['country']=='Italy']
check_q8(reviews.loc[reviews['country']=='Italy'])
# Your code here
reviews.loc[reviews.region_2.notnull()]
check_q9(reviews.loc[reviews.region_2.notnull()])
# Your code here
reviews.loc[:, 'points']
check_q10(reviews.loc[:, 'points'])
# Your code here
reviews.loc[:999, 'points']
check_q11(reviews.loc[:999, 'points'])
# Your code here
reviews.points.iloc[-1000:]
check_q12(reviews.points.iloc[-1000:])
# Your code here
reviews.points.loc[(reviews.country == 'Italy')]
check_q13(reviews.points.loc[(reviews.country == 'Italy')])
# Your code here
reviews.country.loc[((reviews.country == 'Italy')
                    | (reviews.country  == 'France')) & (reviews.points >= 90)]
check_q14(reviews.country.loc[((reviews.country == 'Italy')
                    | (reviews.country  == 'France')) & (reviews.points >= 90)])