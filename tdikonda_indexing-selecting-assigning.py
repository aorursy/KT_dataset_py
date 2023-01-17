import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
df = reviews['description']
df
#check_q1(df)
df = reviews['description'][0]
df
#check_q2(df)
df = reviews.iloc[0]
df
#check_q3(df)
df = reviews.iloc[:10,1]
df
#check_q4(df)
df = reviews.iloc[[1,2,3,5,8]]
df
#check_q5(df)
df = reviews.iloc[[0,1,10,100],[0,5,6,7]]
df
#check_q6(df)
df = reviews.iloc[:100,[0,11]]
df
#check_q7(df)
df = reviews.loc[reviews.country == 'Italy']
df
#check_q8(df)
df = reviews.loc[reviews.region_2.notnull()]
df
#check_q9(df)
df = reviews.points
check_q10(df)
df = reviews.points[:1000]
check_q11(df)
df = reviews.points[-1000:]
check_q12(df)
df = reviews.points[(reviews.country == 'Italy')]
df
check_q13(df)
df = reviews.country[(reviews.country.isin(['Italy', 'France'])) & (reviews.points >= 90)]
df
check_q14(df)