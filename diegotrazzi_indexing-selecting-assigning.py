import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
check_q1(reviews['description'])
reviews['description'][0]
check_q2(reviews['description'][0])
df = reviews.iloc[0,:]
print(df)
check_q3(df)
df = reviews.iloc[0:10,1]
check_q4(reviews.iloc[0:10,1])
df = reviews.iloc[[1,2,3,5,8]]
check_q5(df)
df = reviews.loc[[0,1,10,100],['country','province','region_1','region_2']]
print(df)
check_q6(df)
df = reviews.loc[:100,['country','variety']]
print(df)
check_q7(df)
df = reviews.loc[reviews['country'] == 'Italy']
print(df)
check_q8(df)
df = reviews.loc[reviews['region_2'].notnull()]
print(df)
check_q9(df)
df = reviews['points']
print(df.head(3))
check_q10(df)
# df = reviews['points'].loc[:1000]
df = reviews.iloc[:1000].points
print(df)
check_q11(df)
df = reviews.iloc[-1000:].points
print(df)
check_q12(df)
df = reviews.loc[reviews['country'] == 'Italy']['points']
check_q13(df)
df = reviews.loc[reviews['country'].isin(['Italy','France']) & (reviews['points']>=90)]['country']
print(df)
check_q14(df)