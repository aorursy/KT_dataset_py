import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
check_q1(reviews.loc[:,'description'])
reviews.loc[:,'description']
check_q2(reviews.loc[0,'description'])
check_q3(reviews.iloc[0,:])
check_q4(pd.Series(reviews.loc[0:9,'description']))

check_q5(reviews.iloc[[1,2,3,5,8]])
reviews.loc[[0, 1, 10, 100], ['country', 'province', 'region_1', 'region_2']]
# Your code here
reviews.loc[0:99, ['country', 'variety']]
#answer_q7()
# Your code here
reviews.loc[reviews.country == 'Italy']
# Your code here
reviews.loc[pd.notna(reviews.region_2) ]
# Your code here
reviews.loc[:,'points'].plot.hist()
reviews.loc[0:999,'points'].plot.hist()
l=len(reviews)
reviews.iloc[-1000:,3]
#reviews.loc[(l-1000):l,'points'].plot.hist()
#answer_q12()
reviews.loc[reviews.country == 'Italy']

reviews[reviews.country.isin(["Italy", "France"]) & (reviews.points >= 90)].country