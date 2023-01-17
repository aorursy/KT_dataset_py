import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5)
reviews.head()
check_q1(pd.DataFrame())
# Your code here
d1=reviews.description
check_q1(d1)
# Your code here
d2=reviews['description'][0]
check_q2(d2)
# Your code here
d3=reviews.iloc[0]
check_q3(d3)
# Your code here
d4=reviews.iloc[0:10, 1]
check_q4(d4)
# Your code here
d5=reviews.iloc[[1,2,3,5,8], :]
check_q5(d5)
# Your code here
d6=reviews.loc[[0,1,10,100], ['country', 'province', 'region_1', 'region_2']]
check_q6(d6)
# Your code here
d7=reviews.loc[0:99, ['country', 'variety']]
check_q7(d7)
# Your code here
d8=reviews.loc[reviews.country == 'Italy']
check_q8(d8)
# Your code here
d9=reviews.loc[reviews.region_2.notnull()]
check_q9(d9)
# Your code here
d10=reviews.points
check_q10(d10)
# Your code here
d11=reviews.loc[:999, 'points']
check_q11(d11)
# Your code here
d12=reviews.iloc[-1000:, 3]
check_q12(d12)
# Your code here
d13=reviews[reviews.country == 'Italy'].points
check_q13(d13)
# Your code here
d14=reviews[(reviews.country.isin(['France', 'Italy'])) & (reviews.points >= 90)].country
check_q14(d14)