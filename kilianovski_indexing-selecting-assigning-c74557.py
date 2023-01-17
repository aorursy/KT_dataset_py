import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.country.head()
check_q1(pd.DataFrame())
r = reviews.description
check_q1(r)
r = reviews.description[0]
check_q2(r)
r = reviews.iloc[0]
check_q3(r)
r = reviews.description.iloc[:10]
check_q4(r)
r = reviews.iloc[[1, 2, 3, 5, 8]]
check_q5(r)
r = reviews.loc[[0, 1, 10, 100], ['country', 'province', 'region_1', 'region_2']]
check_q6(r)
r = reviews.loc[:100, ['country', 'variety']] 
check_q7(r) # wtf
r = reviews.loc[reviews.country == 'Italy']
check_q8(r)
r = reviews.loc[reviews.region_2.notnull()]
check_q9(r)

check_q10(reviews.points)
check_q11(reviews.loc[:1000, 'points'])
check_q11(reviews.loc[-1000:, 'points'])
r = reviews.loc[reviews.country == 'Italy', 'points']
check_q13(r)
r = reviews.loc[((reviews.country == 'Italy') | (reviews.country == 'France')) & (reviews.points >= 90), 'country']
check_q14(r)