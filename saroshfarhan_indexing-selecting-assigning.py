import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
selct_desc = reviews.description
selct_desc
check_q1(selct_desc)

# Your code here
check_q2(reviews.description[0])

# Your code here
check_q3(reviews.loc[0])
# Your code here
a = reviews.loc[:9, 'description']
a = pd.Series(a)

check_q4(a)
# Your code here
a = reviews.loc[(reviews.index == 1) | (reviews.index == 2) | (reviews.index == 3) | (reviews.index == 5) | (reviews.index == 8) ]
a
check_q5(a)
# Your code here

check_q6(reviews.loc[[0,1,10,100], ['country', 'province', 'region_1', 'region_2']])

# Your code here
check_q7(reviews.loc[:100, ['country', 'variety']])
# Your code here
check_q8(reviews.loc[reviews.country=='Italy'])
# Your code here
check_q9(reviews.loc[reviews.region_2.notnull()])
# Your code here
check_q10(reviews.points)
# Your code here
check_q11(reviews.loc[:999, 'points'])
# Your code here
check_q12(reviews.iloc[-1000:, 3])

# Your code here
check_q13(reviews.loc[reviews.country=='Italy', 'points'])
# Your code here
check_q14(reviews[reviews.country.isin(["Italy", "France"]) & (reviews.points >= 90)].country)