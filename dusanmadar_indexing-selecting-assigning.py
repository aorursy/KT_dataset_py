import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
check_q1(reviews['description'])
check_q2(reviews.loc[[0], ['description']])
reviews.loc[[0], ['description']]
check_q3(reviews.iloc[0])
check_q4(reviews.iloc[:10, 1])
check_q4(reviews.iloc[:10]['description'])
check_q4(reviews.loc[0:9, 'description'])
check_q5(reviews.loc[[1,2,3,5,8]])
check_q5(reviews.iloc[[1,2,3,5,8]])
a6_1 = reviews.loc[[0,1,10,100], ['country', 'province',  'region_1', 'region_2']]
check_q6(a6_1)
a6_2 = reviews.iloc[[0,1,10,100], [0, 5, 6, 7]]
check_q6(a6_2)
a7_1 = reviews.loc[:99, ['country', 'variety']]
check_q7(a7_1)
a7_2 = reviews.iloc[:100, [0, -2]]
check_q7(a7_2)
a8 = reviews.loc[reviews.country == 'Italy']
check_q8(a8)
df = reviews[reviews.country == "Italy"]
print(df)
print(check_q8(df))
a9 = reviews.loc[reviews.region_2.notnull()]
check_q9(a9)
a10 = reviews.points
check_q10(a10)
a11 = reviews.points[:1000]
check_q11(a11)
a12 = reviews.points[-1000:]
check_q12(a12)
a13 = reviews[reviews.country == 'Italy'].points
check_q13(a13)
a14 = reviews[reviews.country.isin(['Italy', 'France']) & (reviews.points >= 90)].country
check_q14(a14)