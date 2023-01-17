import pandas as pd
import seaborn as sns

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
reviews.description
print(check_q1(reviews.description))
print('-------')
print(answer_q1())

reviews.description[0]
print(check_q2(reviews.description[0]))
print('-------')
print(answer_q2())
reviews.iloc[0]
print(check_q3(reviews.iloc[0]))
print('-------')
print(answer_q3())
reviews.iloc[0:10, 1]
print(check_q4(reviews.iloc[0:10, 1]))
print('-------')
print(answer_q4())
reviews.iloc[[1,2,3,5,8]]
print(check_q5(reviews.iloc[[1,2,3,5,8]]))
print('-------')
print(answer_q5())
reviews.loc[[0, 1, 10, 100], ['country', 'province', 'region_1', 'region_2']]
print(check_q6(reviews.loc[[0, 1, 10, 100], ['country', 'province', 'region_1', 'region_2']]))
print('-------')
print(answer_q6())
reviews.loc[0:100, ['country', 'variety']]
print(check_q7(reviews.loc[0:100, ['country', 'variety']]))
print('-------')
print(answer_q7())
reviews.loc[reviews.country == 'Italy']
print(check_q8(reviews.loc[reviews.country == 'Italy']))
print('-------')
print(answer_q8())
reviews.loc[reviews.region_2.notnull()]
print(check_q9(reviews.loc[reviews.region_2.notnull()]))
print('-------')
print(answer_q9())
reviews.points
print(check_q10(reviews.points))
print(answer_q10())
reviews.loc[:1000, 'points']
print(check_q11(reviews.loc[:1000, 'points']))
print(answer_q11())
reviews.iloc[-1000:, 3]
print(check_q12(reviews.iloc[-1000:, 3]))
print(answer_q12())
reviews[reviews.country == "Italy"].points
check_q13(reviews[reviews.country == "Italy"].points)
print(answer_q13())
reviews[reviews.country.isin(["Italy", "France"]) & (reviews.points >= 90)].country
check_q14(reviews[reviews.country.isin(["Italy", "France"]) & (reviews.points >= 90)].country)
print(answer_q14())