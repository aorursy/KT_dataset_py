import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
print(reviews['description'])
check_q1(reviews['description'])

print(reviews['description'][0])
check_q2(reviews['description'][0])
print(reviews.loc[0])
check_q3(reviews.loc[0])
print(pd.Series(reviews['description'][0:10]))
check_q4(pd.Series(reviews['description'][0:10]))
r = [1,2,3,5,8]
check_q5(reviews.iloc[l])
r = [0,1,10,100]
c = ['country', 'province', 'region_1', 'region_2']
check_q6(reviews[c].iloc[r])
c = ['country', 'variety']
check_q7(reviews[c].iloc[0:100])
check_q8(reviews[reviews.country == 'Italy'])
check_q9(reviews[reviews.region_2.notnull()])
check_q10(pd.Series(reviews['points']))
check_q11(pd.Series(reviews.points[0:1000]))
check_q12(pd.Series(reviews.points.iloc[-1000:]))
check_q13(pd.Series(reviews.points[reviews.country == 'Italy']))
check_q14(pd.Series(reviews.country[((reviews.country == 'Italy') | (reviews.country == 'France')) & (reviews.points > 90)]))
