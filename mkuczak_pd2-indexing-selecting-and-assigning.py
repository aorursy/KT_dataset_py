import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
df1 = reviews.description
check_q1(df1)
df2 = reviews.description[0]
check_q2(df2)
df3 = reviews.iloc[0]
check_q3(df3)
df4 = reviews.description[0:10]
check_q4(df4)
df5 = reviews.iloc[[1,2,3,5,8]]
check_q5(df5)
df6 = reviews.loc[[0,1,10,100], ['country', 'province', 'region_1', 'region_2']]
check_q6(df6)
df7 = reviews.loc[0:100,['country','variety']]
check_q7(df7)
df8 = reviews.loc[(reviews.country == 'Italy')]
check_q8(df8)
df9 = reviews.loc[reviews.region_2.notnull()]
check_q9(df9)
df10 = reviews.points
check_q10(df10)
df11 = reviews.loc[:999, 'points']
check_q11(df11)
df12 = reviews.iloc[-1000:, 3]
check_q12(df12)
df13 = reviews[reviews.country == 'Italy'].points
check_q13(df13)
df14 = reviews[((reviews.country == 'Italy') | (reviews.country == 'France')) & (reviews.points >= 90)].country
check_q14(df14)