import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
print(answer_q1)
# Your code here
desccription = reviews['description']
check_q1(desccription)
# Your code here
value1 = reviews.loc[0, 'description']
check_q2(value1)
# Your code here
row1 = reviews.loc[0]
check_q3(row1)
# Your code here
values10 = reviews.loc[0:9, 'description']
check_q4(values10)
# Your code here
some_rows = reviews.loc[[1, 2, 3, 5, 8]]
check_q5(some_rows)
# Your code here
reviews.loc[[0, 1, 10, 100], ['country', 'province', 'region_1', 'region_2']]
# Your code here
E7 = reviews[['country', 'variety']].iloc[:100]
check_q7(E7)
# Your code here
E8 = reviews.loc[reviews.country == 'Italy']
check_q8(E8)
# Your code here
E9 = reviews.loc[reviews.region_2.notnull()]
check_q9(E9)
# Your code here
E10 = reviews.points
check_q10(E10)
# Your code here
E11 = reviews.points.iloc[:1000]
check_q11(E11)
# Your code here
E12 = reviews.points.iloc[-1000:]
check_q12(E12)
# Your code here
E13 = reviews.points[reviews.country == 'Italy']
check_q13(E13)
# Your code here
E14 = reviews[reviews.country.isin(['Italy', 'France']) & (reviews.points >= 90)].country
check_q14(E14)