import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
a=reviews.description
check_q1(a)
b=reviews.description[0]
check_q2(b)
c=reviews.iloc[0]
check_q3(c)
d=reviews.description.iloc[:10]
check_q4(d)
e=reviews.iloc[[1,2,3,5,8]]
check_q5(e)
f=reviews.loc[[0, 1, 10, 100],['country', 'province', 'region_1', 'region_2']]
check_q6(f)
g=reviews.loc[:100,['country', 'variety']]
check_q7(g)
h=reviews.loc[reviews.country=='Italy']
check_q8(h)
i=reviews.loc[reviews.region_2.notnull()]
check_q9(i)
j=reviews['points']
check_q10(j)
k=reviews['points'][:1000]
check_q11(k)
l=reviews['points'][-1000:]
check_q12(l)
m=reviews.loc[reviews.country.isin(['Italy'])]['points']
check_q13(m)
n=reviews.loc[(reviews.country.isin(['Italy', 'France'])) & (reviews.points >= 90)]['country']
check_q14(n)