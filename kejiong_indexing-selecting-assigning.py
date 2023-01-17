import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
pd.DataFrame(reviews, columns =['description'])
reviews['description']
check_q1(reviews['description'])
reviews.loc[0,'description']
check_q2(reviews.loc[0,'description'])
reviews.loc[[0]]
check_q3(reviews.loc[0])
data = reviews.loc[:9,'description']
df = pd.Series(data)
print(df)
check_q4(df)
df = reviews.iloc[[1,2,3,5,8]]
print(df)
check_q5(df)
df = reviews.loc[[0,1,10,100],['country', 'province','region_1','region_2']]
check_q6(df)
df = reviews.loc[0:100,['country','variety']]
print(df)
check_q7(df)
reviews[reviews.country=='Italy']
reviews[reviews.region_2.notnull()] 
reviews.points
reviews.points[:1000]
reviews.points[-1000:]
reviews[reviews.country =='Italy'].points
reviews[reviews.country.isin(['France','Italy']) & (reviews.points >=90)].country